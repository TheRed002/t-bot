"""Service interfaces for the utils module."""

from abc import abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.core.base import BaseService

if TYPE_CHECKING:
    from src.utils.validation.service import ValidationContext, ValidationResult


@runtime_checkable
class ValidationServiceInterface(Protocol):
    """Interface for validation services."""

    @abstractmethod
    async def validate_order(
        self, order_data: dict[str, Any], context: "ValidationContext | None" = None
    ) -> "ValidationResult":
        """Validate trading order data."""
        ...

    @abstractmethod
    async def validate_risk_parameters(
        self, risk_data: dict[str, Any], context: "ValidationContext | None" = None
    ) -> "ValidationResult":
        """Validate risk management parameters."""
        ...

    @abstractmethod
    async def validate_strategy_config(
        self, strategy_data: dict[str, Any], context: "ValidationContext | None" = None
    ) -> "ValidationResult":
        """Validate strategy configuration."""
        ...

    @abstractmethod
    async def validate_market_data(
        self, market_data: dict[str, Any], context: "ValidationContext | None" = None
    ) -> "ValidationResult":
        """Validate market data."""
        ...

    @abstractmethod
    async def validate_batch(
        self, validations: list[tuple[str, Any]], context: "ValidationContext | None" = None
    ) -> dict[str, "ValidationResult"]:
        """Validate multiple items in batch."""
        ...


@runtime_checkable
class GPUInterface(Protocol):
    """Interface for GPU management services."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if GPU is available."""
        ...

    @abstractmethod
    def get_memory_info(self) -> dict[str, Any]:
        """Get GPU memory information."""
        ...


@runtime_checkable
class PrecisionInterface(Protocol):
    """Interface for precision tracking services."""

    @abstractmethod
    def track_operation(self, operation: str, input_precision: int, output_precision: int) -> None:
        """Track precision changes during operations."""
        ...

    @abstractmethod
    def get_precision_stats(self) -> dict[str, Any]:
        """Get precision tracking statistics."""
        ...


@runtime_checkable
class DataFlowInterface(Protocol):
    """Interface for data flow validation services."""

    @abstractmethod
    def validate_data_integrity(self, data: Any) -> bool:
        """Validate data integrity."""
        ...

    @abstractmethod
    def get_validation_report(self) -> dict[str, Any]:
        """Get validation report."""
        ...


@runtime_checkable
class CalculatorInterface(Protocol):
    """Interface for financial calculation services."""

    @abstractmethod
    def calculate_compound_return(self, principal: Decimal, rate: Decimal, periods: int) -> Decimal:
        """Calculate compound return."""
        ...

    @abstractmethod
    def calculate_sharpe_ratio(self, returns: list[Decimal], risk_free_rate: Decimal) -> Decimal:
        """Calculate Sharpe ratio."""
        ...


class BaseUtilityService(BaseService):
    """Base class for utility services that inherits from core BaseService."""

    def __init__(self, name: str | None = None, config: dict[str, Any] | None = None):
        """Initialize base utility service."""
        super().__init__(name or self.__class__.__name__)
        self.config = config or {}

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service."""
        ...


__all__ = [
    "BaseUtilityService",
    "CalculatorInterface",
    "DataFlowInterface",
    "GPUInterface",
    "PrecisionInterface",
    "ValidationServiceInterface",
]
