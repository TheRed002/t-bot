"""
Capital Management Service Interfaces

This module defines the abstract protocols and interfaces for all capital management
services to ensure proper service layer architecture and contract enforcement.

All interfaces are infrastructure-agnostic and focus on business operations,
not implementation details like databases or external APIs.

Version: 2.0.0 - Production Ready
Author: Trading Bot Framework
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Protocol

from src.core.types.capital import (
    CapitalCurrencyExposure,
    CapitalExchangeAllocation,
    CapitalFundFlow,
)
from src.core.types.risk import CapitalAllocation, CapitalMetrics


# Infrastructure-agnostic repository protocols
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


class ExchangeDataServiceProtocol(Protocol):
    """Protocol for exchange data service operations."""

    async def get_tickers(self) -> dict[str, Any]: ...
    async def get_order_book(
        self, exchange: str, symbol: str, limit: int = 50
    ) -> dict[str, Any]: ...
    async def get_status(self, exchange: str) -> dict[str, Any]: ...
    async def get_fees(self, exchange: str) -> dict[str, Any]: ...


class CapitalServiceProtocol(Protocol):
    """Protocol for capital management service operations."""

    async def allocate_capital(
        self,
        strategy_id: str,
        exchange: str,
        requested_amount: Decimal,
        bot_id: str | None = None,
        authorized_by: str | None = None,
        risk_context: dict[str, Any] | None = None,
    ) -> CapitalAllocation:
        """Allocate capital to a strategy with full audit trail.

        This is a pure business operation that should not depend on
        specific database implementations or external APIs.
        """
        ...

    async def release_capital(
        self,
        strategy_id: str,
        exchange: str,
        release_amount: Decimal,
        bot_id: str | None = None,
        authorized_by: str | None = None,
    ) -> bool:
        """Release allocated capital with full audit trail."""
        ...

    async def update_utilization(
        self,
        strategy_id: str,
        exchange: str,
        utilized_amount: Decimal,
        bot_id: str | None = None,
    ) -> bool:
        """Update capital utilization for a strategy."""
        ...

    async def get_capital_metrics(self) -> CapitalMetrics:
        """Get current capital management metrics."""
        ...

    async def get_allocations_by_strategy(self, strategy_id: str) -> list[CapitalAllocation]:
        """Get all capital allocations for a specific strategy."""
        ...


class CurrencyManagementServiceProtocol(Protocol):
    """Protocol for currency management service operations."""

    async def update_currency_exposures(
        self, balances: dict[str, dict[str, Decimal]]
    ) -> dict[str, CapitalCurrencyExposure]:
        """Update currency exposures based on current balances.

        This operation processes balance data and calculates exposures
        without depending on specific exchange implementations.
        """
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
        """Distribute capital across exchanges based on optimization criteria.

        This is a business logic operation that should work with exchange
        abstractions, not specific exchange implementations.
        """
        ...

    async def rebalance_exchanges(self) -> dict[str, CapitalExchangeAllocation]:
        """Rebalance capital across exchanges based on current metrics."""
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
        self, amount: Decimal, currency: str = "USDT", exchange: str = "binance"
    ) -> CapitalFundFlow:
        """Process a deposit request.

        This operation handles deposit business logic including validation
        and flow tracking without depending on specific infrastructure.
        """
        ...

    async def process_withdrawal(
        self,
        amount: Decimal,
        currency: str = "USDT",
        exchange: str = "binance",
        reason: str = "withdrawal",
    ) -> CapitalFundFlow:
        """Process a withdrawal request with rule validation."""
        ...

    async def process_strategy_reallocation(
        self, from_strategy: str, to_strategy: str, amount: Decimal, reason: str = "reallocation"
    ) -> CapitalFundFlow:
        """Process capital reallocation between strategies."""
        ...

    async def get_flow_history(self, days: int = 30) -> list[CapitalFundFlow]:
        """Get fund flow history for the specified period."""
        ...


# Abstract base classes for service implementations


class AbstractCapitalService(ABC):
    """Abstract base class for capital service implementations."""

    @abstractmethod
    async def allocate_capital(
        self,
        strategy_id: str,
        exchange: str,
        requested_amount: Decimal,
        bot_id: str | None = None,
        authorized_by: str | None = None,
        risk_context: dict[str, Any] | None = None,
    ) -> CapitalAllocation:
        """Allocate capital to a strategy with full audit trail."""
        pass

    @abstractmethod
    async def release_capital(
        self,
        strategy_id: str,
        exchange: str,
        release_amount: Decimal,
        bot_id: str | None = None,
        authorized_by: str | None = None,
    ) -> bool:
        """Release allocated capital with full audit trail."""
        pass

    @abstractmethod
    async def get_capital_metrics(self) -> CapitalMetrics:
        """Get current capital management metrics."""
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
        """Distribute capital across exchanges based on optimization criteria."""
        pass

    @abstractmethod
    async def rebalance_exchanges(self) -> dict[str, CapitalExchangeAllocation]:
        """Rebalance capital across exchanges based on current metrics."""
        pass


class AbstractFundFlowManagementService(ABC):
    """Abstract base class for fund flow management service implementations."""

    @abstractmethod
    async def process_deposit(
        self, amount: Decimal, currency: str = "USDT", exchange: str = "binance"
    ) -> CapitalFundFlow:
        """Process a deposit request."""
        pass

    @abstractmethod
    async def process_withdrawal(
        self,
        amount: Decimal,
        currency: str = "USDT",
        exchange: str = "binance",
        reason: str = "withdrawal",
    ) -> CapitalFundFlow:
        """Process a withdrawal request with rule validation."""
        pass
