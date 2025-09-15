"""
Capital Management Service Interfaces.

Simple interfaces for capital management services.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Protocol

from src.capital_management.constants import DEFAULT_BASE_CURRENCY, DEFAULT_EXCHANGE
from src.core.types import CapitalAllocation, CapitalMetrics
from src.core.types.capital import (
    CapitalCurrencyExposure,
    CapitalExchangeAllocation,
    CapitalFundFlow,
)


# Repository protocols
class CapitalRepositoryProtocol(Protocol):
    """Protocol for capital allocation repository operations."""

    async def create(self, allocation_data: dict[str, Any]) -> Any: ...

    async def update(self, allocation_data: dict[str, Any]) -> Any: ...

    async def delete(self, allocation_id: str) -> bool: ...

    async def get_by_strategy_exchange(self, strategy_id: str, exchange: str) -> Any | None: ...

    async def get_by_strategy(self, strategy_id: str) -> list[Any]: ...

    async def get_all(self, limit: int | None = None) -> list[Any]: ...


class AuditRepositoryProtocol(Protocol):
    """Protocol for audit log repository operations."""

    async def create(self, audit_data: dict[str, Any]) -> Any: ...


# Service protocols
class CapitalServiceProtocol(Protocol):
    """Protocol for capital management service operations."""

    async def allocate_capital(
        self,
        strategy_id: str,
        exchange: str,
        amount: Decimal,
        min_allocation: Decimal | None = None,
        max_allocation: Decimal | None = None,
        target_allocation_pct: Decimal | None = None,
        authorized_by: str | None = None,
    ) -> CapitalAllocation:
        """Allocate capital to a strategy."""
        ...

    async def release_capital(
        self,
        strategy_id: str,
        exchange: str,
        amount: Decimal,
        authorized_by: str | None = None,
    ) -> bool:
        """Release allocated capital."""
        ...

    async def update_utilization(
        self,
        strategy_id: str,
        exchange: str,
        utilized_amount: Decimal,
        authorized_by: str | None = None,
    ) -> bool:
        """Update capital utilization for a strategy."""
        ...

    async def get_capital_metrics(self) -> CapitalMetrics:
        """Get current capital management metrics."""
        ...

    async def get_allocations_by_strategy(self, strategy_id: str) -> list[CapitalAllocation]:
        """Get all capital allocations for a specific strategy."""
        ...

    async def get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]:
        """Get all capital allocations across all strategies."""
        ...


class CapitalAllocatorProtocol(Protocol):
    """Protocol for capital allocator operations."""

    async def allocate_capital(
        self,
        strategy_id: str,
        exchange: str,
        requested_amount: Decimal,
        **kwargs,
    ) -> CapitalAllocation:
        """Allocate capital to a strategy."""
        ...

    async def release_capital(
        self,
        strategy_id: str,
        exchange: str,
        release_amount: Decimal,
        **kwargs,
    ) -> bool:
        """Release allocated capital."""
        ...

    async def get_capital_metrics(self) -> CapitalMetrics:
        """Get current capital metrics."""
        ...


class CurrencyManagementServiceProtocol(Protocol):
    """Protocol for currency management service operations."""

    async def update_currency_exposures(
        self, balances: dict[str, dict[str, Decimal]]
    ) -> dict[str, CapitalCurrencyExposure]:
        """Update currency exposures based on current balances."""
        ...

    async def calculate_hedging_requirements(self) -> dict[str, Decimal]:
        """Calculate hedging requirements for currency exposures."""
        ...

    async def execute_currency_conversion(
        self, from_currency: str, to_currency: str, amount: Decimal, exchange: str
    ) -> CapitalFundFlow:
        """Execute currency conversion between currencies."""
        ...

    async def get_currency_risk_metrics(self) -> dict[str, dict[str, float]]:
        """Calculate currency risk metrics."""
        ...


class ExchangeDistributionServiceProtocol(Protocol):
    """Protocol for exchange distribution service operations."""

    async def distribute_capital(
        self, total_amount: Decimal
    ) -> dict[str, CapitalExchangeAllocation]:
        """Distribute capital across exchanges."""
        ...

    async def rebalance_exchanges(self) -> dict[str, CapitalExchangeAllocation]:
        """Rebalance capital across exchanges."""
        ...

    async def get_exchange_allocation(self, exchange: str) -> CapitalExchangeAllocation | None:
        """Get current allocation for a specific exchange."""
        ...

    async def update_exchange_utilization(self, exchange: str, utilized_amount: Decimal) -> None:
        """Update utilization for a specific exchange."""
        ...


class FundFlowManagementServiceProtocol(Protocol):
    """Protocol for fund flow management service operations."""

    async def process_deposit(
        self,
        amount: Decimal,
        currency: str = DEFAULT_BASE_CURRENCY,
        exchange: str = DEFAULT_EXCHANGE,
    ) -> CapitalFundFlow:
        """Process a deposit request."""
        ...

    async def process_withdrawal(
        self,
        amount: Decimal,
        currency: str = DEFAULT_BASE_CURRENCY,
        exchange: str = DEFAULT_EXCHANGE,
        reason: str = "withdrawal",
    ) -> CapitalFundFlow:
        """Process a withdrawal request."""
        ...

    async def process_strategy_reallocation(
        self, from_strategy: str, to_strategy: str, amount: Decimal, reason: str = "reallocation"
    ) -> CapitalFundFlow:
        """Process capital reallocation between strategies."""
        ...

    async def get_flow_history(self, days: int = 30) -> list[CapitalFundFlow]:
        """Get fund flow history for the specified period."""
        ...


# Abstract base classes
class AbstractCapitalService(ABC):
    """Abstract base class for capital service implementations."""

    @abstractmethod
    async def allocate_capital(
        self,
        strategy_id: str,
        exchange: str,
        amount: Decimal,
        min_allocation: Decimal | None = None,
        max_allocation: Decimal | None = None,
        target_allocation_pct: Decimal | None = None,
        authorized_by: str | None = None,
    ) -> CapitalAllocation:
        """Allocate capital to a strategy."""
        pass

    @abstractmethod
    async def release_capital(
        self,
        strategy_id: str,
        exchange: str,
        amount: Decimal,
        authorized_by: str | None = None,
    ) -> bool:
        """Release allocated capital."""
        pass

    @abstractmethod
    async def get_capital_metrics(self) -> CapitalMetrics:
        """Get current capital management metrics."""
        pass

    @abstractmethod
    async def get_all_allocations(self, limit: int | None = None) -> list[CapitalAllocation]:
        """Get all capital allocations."""
        pass


class AbstractCurrencyManagementService(ABC):
    """Abstract base class for currency management service implementations."""

    @abstractmethod
    async def update_currency_exposures(
        self, balances: dict[str, dict[str, Decimal]]
    ) -> dict[str, CapitalCurrencyExposure]:
        """Update currency exposures based on current balances."""
        pass

    @abstractmethod
    async def calculate_hedging_requirements(self) -> dict[str, Decimal]:
        """Calculate hedging requirements for currency exposures."""
        pass


class AbstractExchangeDistributionService(ABC):
    """Abstract base class for exchange distribution service implementations."""

    @abstractmethod
    async def distribute_capital(
        self, total_amount: Decimal
    ) -> dict[str, CapitalExchangeAllocation]:
        """Distribute capital across exchanges."""
        pass

    @abstractmethod
    async def rebalance_exchanges(self) -> dict[str, CapitalExchangeAllocation]:
        """Rebalance capital across exchanges."""
        pass


class AbstractFundFlowManagementService(ABC):
    """Abstract base class for fund flow management service implementations."""

    @abstractmethod
    async def process_deposit(
        self,
        amount: Decimal,
        currency: str = DEFAULT_BASE_CURRENCY,
        exchange: str = DEFAULT_EXCHANGE,
    ) -> CapitalFundFlow:
        """Process a deposit request."""
        pass

    @abstractmethod
    async def process_withdrawal(
        self,
        amount: Decimal,
        currency: str = DEFAULT_BASE_CURRENCY,
        exchange: str = DEFAULT_EXCHANGE,
        reason: str = "withdrawal",
    ) -> CapitalFundFlow:
        """Process a withdrawal request."""
        pass
