"""Service interfaces for the utils module."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.core.types.base import ConfigDict

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


class BaseUtilityService(ABC):
    """Base class for utility services."""

    def __init__(self, name: str | None = None, config: ConfigDict | None = None):
        """Initialize base utility service."""
        self.name = name or self.__class__.__name__
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
    "ValidationServiceInterface",
]
