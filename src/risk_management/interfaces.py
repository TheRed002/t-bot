"""
Risk Management Service Interfaces.

This module defines the core interfaces and protocols for the risk management
service layer, ensuring proper separation of concerns and dependency injection.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from src.core.types import (
    MarketData,
    OrderRequest,
    Position,
    PositionSizeMethod,
    RiskLevel,
    RiskMetrics,
    Signal,
)


@runtime_checkable
class CacheServiceInterface(Protocol):
    """Protocol for cache service implementations."""

    async def get(self, key: str) -> Any:
        """Get value from cache."""
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with optional TTL."""
        ...

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        ...

    async def clear(self) -> None:
        """Clear all cache entries."""
        ...

    async def close(self) -> None:
        """Close cache connections."""
        ...


@runtime_checkable
class ExchangeServiceInterface(Protocol):
    """Protocol for exchange service implementations to avoid direct coupling."""

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all orders for symbol or all orders if symbol is None."""
        ...

    async def close_all_positions(self) -> int:
        """Close all open positions."""
        ...

    async def get_account_balance(self) -> Decimal:
        """Get current account balance."""
        ...


@runtime_checkable
class RiskServiceInterface(Protocol):
    """Protocol for risk management service implementations."""

    async def calculate_position_size(
        self,
        signal: Signal,
        available_capital: Decimal,
        current_price: Decimal,
        method: PositionSizeMethod | None = None,
    ) -> Decimal:
        """Calculate optimal position size."""
        ...

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate trading signal against risk constraints."""
        ...

    async def validate_order(self, order: OrderRequest) -> bool:
        """Validate order against risk constraints."""
        ...

    async def calculate_risk_metrics(
        self, positions: list[Position] | None = None, market_data: list[MarketData] | None = None
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        ...

    async def should_exit_position(self, position: Position, market_data: MarketData) -> bool:
        """Determine if position should be closed."""
        ...

    def get_current_risk_level(self) -> RiskLevel:
        """Get current risk level."""
        ...

    def is_emergency_stop_active(self) -> bool:
        """Check if emergency stop is active."""
        ...

    async def get_risk_summary(self) -> dict[str, Any]:
        """Get comprehensive risk summary."""
        ...

    async def get_portfolio_metrics(self) -> Any:
        """Get current portfolio metrics."""
        ...

    async def validate_risk_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Validate risk parameters."""
        ...

    async def get_current_risk_limits(self) -> dict[str, Any]:
        """Get current risk limits configuration."""
        ...


@runtime_checkable
class PositionSizingServiceInterface(Protocol):
    """Protocol for position sizing service implementations."""

    async def calculate_size(
        self,
        signal: Signal,
        available_capital: Decimal,
        current_price: Decimal,
        method: PositionSizeMethod | None = None,
    ) -> Decimal:
        """Calculate position size."""
        ...

    async def validate_size(self, position_size: Decimal, available_capital: Decimal) -> bool:
        """Validate calculated position size."""
        ...


@runtime_checkable
class RiskMetricsServiceInterface(Protocol):
    """Protocol for risk metrics service implementations."""

    async def calculate_metrics(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> RiskMetrics:
        """Calculate risk metrics."""
        ...

    async def get_portfolio_value(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> Decimal:
        """Calculate portfolio value."""
        ...


@runtime_checkable
class RiskValidationServiceInterface(Protocol):
    """Protocol for risk validation service implementations."""

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate signal."""
        ...

    async def validate_order(self, order: OrderRequest) -> bool:
        """Validate order."""
        ...

    async def validate_portfolio_limits(self, new_position: Position) -> bool:
        """Validate portfolio limits."""
        ...


@runtime_checkable
class PortfolioLimitsServiceInterface(Protocol):
    """Protocol for portfolio limits service implementations."""

    async def check_portfolio_limits(self, new_position: Position) -> bool:
        """Check if adding position would violate portfolio limits."""
        ...

    async def update_portfolio_state(
        self, positions: list[Position], portfolio_value: Decimal
    ) -> None:
        """Update portfolio state for limit calculations."""
        ...

    async def update_return_history(self, symbol: str, price: Decimal) -> None:
        """Update return history for correlation calculations."""
        ...

    async def get_portfolio_summary(self) -> dict[str, Any]:
        """Get comprehensive portfolio limits summary."""
        ...


@runtime_checkable
class RiskMonitoringServiceInterface(Protocol):
    """Protocol for risk monitoring service implementations."""

    async def start_monitoring(self, interval: int = 60) -> None:
        """Start risk monitoring."""
        ...

    async def stop_monitoring(self) -> None:
        """Stop risk monitoring."""
        ...

    async def check_emergency_conditions(self, metrics: RiskMetrics) -> bool:
        """Check for emergency stop conditions."""
        ...

    async def get_risk_summary(self) -> dict[str, Any]:
        """Get comprehensive risk summary."""
        ...


class AbstractRiskService(ABC):
    """Abstract base class for risk services."""

    @abstractmethod
    async def start(self) -> None:
        """Start the service."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the service."""
        pass

    @abstractmethod
    async def calculate_position_size(
        self,
        signal: Signal,
        available_capital: Decimal,
        current_price: Decimal,
        method: PositionSizeMethod | None = None,
    ) -> Decimal:
        """Calculate position size."""
        pass

    @abstractmethod
    async def validate_signal(self, signal: Signal) -> bool:
        """Validate signal."""
        pass

    @abstractmethod
    async def calculate_risk_metrics(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> RiskMetrics:
        """Calculate risk metrics."""
        pass


@runtime_checkable
class RiskMetricsRepositoryInterface(Protocol):
    """Protocol for risk metrics data access."""

    async def get_historical_returns(self, symbol: str, days: int) -> list[Decimal]:
        """Get historical returns for symbol."""
        ...

    async def get_price_history(self, symbol: str, days: int) -> list[Decimal]:
        """Get price history for symbol."""
        ...

    async def get_portfolio_positions(self) -> list[Position]:
        """Get current portfolio positions."""
        ...

    async def save_risk_metrics(self, metrics: RiskMetrics) -> None:
        """Save calculated risk metrics."""
        ...

    async def get_correlation_data(self, symbols: list[str], days: int) -> dict[str, list[Decimal]]:
        """Get correlation data for symbols."""
        ...


@runtime_checkable
class PortfolioRepositoryInterface(Protocol):
    """Protocol for portfolio data access."""

    async def get_current_positions(self) -> list[Position]:
        """Get current portfolio positions."""
        ...

    async def get_portfolio_value(self) -> Decimal:
        """Get current portfolio value."""
        ...

    async def get_position_history(self, symbol: str, days: int) -> list[Position]:
        """Get position history for symbol."""
        ...

    async def update_portfolio_limits(self, limits: dict[str, Any]) -> None:
        """Update portfolio limits."""
        ...


class RiskManagementFactoryInterface(ABC):
    """Abstract interface for risk management service factories."""

    @abstractmethod
    def create_risk_service(self, correlation_id: str | None = None) -> "RiskServiceInterface":
        """Create a RiskService instance with dependency injection."""
        pass

    @abstractmethod
    def create_risk_management_controller(self, correlation_id: str | None = None) -> Any:
        """Create a RiskManagementController instance with dependency injection."""
        pass

    @abstractmethod
    def validate_dependencies(self) -> dict[str, bool]:
        """Validate the availability of dependencies."""
        pass
