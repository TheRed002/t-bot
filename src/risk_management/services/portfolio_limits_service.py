"""
Portfolio Limits Service for Risk Management Module.

This service wraps the PortfolioLimits component with proper service layer patterns,
providing business logic encapsulation and dependency injection support.
"""

from decimal import Decimal
from typing import Any

from src.core.base import BaseService
from src.core.config import Config
from src.core.exceptions import ServiceError, ValidationError
from src.core.types import Position
from src.error_handling.decorators import with_error_context
from src.risk_management.interfaces import PortfolioLimitsServiceInterface
from src.risk_management.portfolio_limits import PortfolioLimits
from src.utils.decorators import time_execution


class PortfolioLimitsService(BaseService, PortfolioLimitsServiceInterface):
    """Service layer for portfolio limits enforcement."""

    def __init__(self, config: Config):
        """
        Initialize portfolio limits service.

        Args:
            config: Application configuration
        """
        super().__init__()
        self._config = config
        self._portfolio_limits = PortfolioLimits(config)

    @with_error_context(component="portfolio_limits", operation="check_portfolio_limits")
    @time_execution
    async def check_portfolio_limits(self, new_position: Position) -> bool:
        """
        Check if adding position would violate portfolio limits.

        Args:
            new_position: Position to be added

        Returns:
            bool: True if position addition is allowed

        Raises:
            ServiceError: If limits check fails
            ValidationError: If position data is invalid
        """
        try:
            if not new_position:
                raise ValidationError("Position is required for limits check")

            return await self._portfolio_limits.check_portfolio_limits(new_position)

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(
                "Portfolio limits check failed",
                error=str(e),
                symbol=new_position.symbol if new_position else None,
            )
            raise ServiceError(f"Portfolio limits check failed: {e}") from e

    @with_error_context(component="portfolio_limits", operation="update_portfolio_state")
    @time_execution
    async def update_portfolio_state(
        self, positions: list[Position], portfolio_value: Decimal
    ) -> None:
        """
        Update portfolio state for limit calculations.

        Args:
            positions: Current portfolio positions
            portfolio_value: Current total portfolio value

        Raises:
            ServiceError: If update fails
            ValidationError: If input data is invalid
        """
        try:
            if positions is None:
                positions = []

            if portfolio_value is None or portfolio_value < Decimal("0"):
                raise ValidationError("Portfolio value must be non-negative")

            await self._portfolio_limits.update_portfolio_state(positions, portfolio_value)

            self.logger.info(
                "Portfolio state updated",
                position_count=len(positions),
                portfolio_value=str(portfolio_value),
            )

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(
                "Portfolio state update failed",
                error=str(e),
                position_count=len(positions) if positions else 0,
            )
            raise ServiceError(f"Portfolio state update failed: {e}") from e

    @with_error_context(component="portfolio_limits", operation="update_return_history")
    @time_execution
    async def update_return_history(self, symbol: str, price: Decimal) -> None:
        """
        Update return history for correlation calculations.

        Args:
            symbol: Trading symbol
            price: Current price

        Raises:
            ServiceError: If update fails
            ValidationError: If input data is invalid
        """
        try:
            if not symbol or not symbol.strip():
                raise ValidationError("Symbol is required")

            if not price or price <= Decimal("0"):
                raise ValidationError("Price must be positive")

            await self._portfolio_limits.update_return_history(symbol, price)

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(
                "Return history update failed",
                error=str(e),
                symbol=symbol,
                price=str(price) if price else None,
            )
            raise ServiceError(f"Return history update failed: {e}") from e

    @with_error_context(component="portfolio_limits", operation="get_portfolio_summary")
    @time_execution
    async def get_portfolio_summary(self) -> dict[str, Any]:
        """
        Get comprehensive portfolio limits summary.

        Returns:
            Dict containing current portfolio state and limits

        Raises:
            ServiceError: If summary generation fails
        """
        try:
            return await self._portfolio_limits.get_portfolio_summary()

        except Exception as e:
            self.logger.error("Portfolio summary generation failed", error=str(e))
            raise ServiceError(f"Portfolio summary generation failed: {e}") from e

    async def start(self) -> None:
        """Start the service."""
        self.logger.info("Portfolio limits service started")

    async def stop(self) -> None:
        """Stop the service."""
        self.logger.info("Portfolio limits service stopped")

    async def health_check(self) -> bool:
        """Check service health."""
        try:
            # Basic health check - verify portfolio limits component is working
            test_summary = await self._portfolio_limits.get_portfolio_summary()
            return isinstance(test_summary, dict)
        except Exception as e:
            self.logger.error("Portfolio limits service health check failed", error=str(e))
            return False
