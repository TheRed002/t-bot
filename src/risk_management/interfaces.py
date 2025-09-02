"""
Risk Management Service Interfaces.

This module defines the core interfaces and protocols for the risk management
service layer, ensuring proper separation of concerns and dependency injection.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Protocol

from src.core.types import (
    MarketData,
    OrderRequest,
    Position,
    RiskLevel,
    RiskMetrics,
    Signal,
)


class RiskServiceInterface(Protocol):
    """Protocol for risk management service implementations."""

    async def calculate_position_size(
        self,
        signal: Signal,
        available_capital: Decimal,
        current_price: Decimal,
        method: str | None = None,
    ) -> Decimal:
        """Calculate optimal position size."""
        ...

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate trading signal against risk constraints."""
        ...

    async def validate_order(self, order: OrderRequest) -> bool:
        """Validate order against risk constraints."""
        ...

    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics:
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


class PositionSizingServiceInterface(Protocol):
    """Protocol for position sizing service implementations."""

    async def calculate_size(
        self,
        signal: Signal,
        available_capital: Decimal,
        current_price: Decimal,
        method: str | None = None,
    ) -> Decimal:
        """Calculate position size."""
        ...

    async def validate_size(self, position_size: Decimal, available_capital: Decimal) -> bool:
        """Validate calculated position size."""
        ...


class RiskMetricsServiceInterface(Protocol):
    """Protocol for risk metrics service implementations."""

    async def calculate_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics:
        """Calculate risk metrics."""
        ...

    async def get_portfolio_value(self, positions: list[Position], market_data: list[MarketData]) -> Decimal:
        """Calculate portfolio value."""
        ...


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
        method: str | None = None,
    ) -> Decimal:
        """Calculate position size."""
        pass

    @abstractmethod
    async def validate_signal(self, signal: Signal) -> bool:
        """Validate signal."""
        pass

    @abstractmethod
    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics:
        """Calculate risk metrics."""
        pass


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
