"""
Risk Manager Adapter for Execution Algorithms.

This adapter bridges the interface mismatch between what execution algorithms
expect (risk_manager.validate_order) and what RiskService provides.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.exceptions import (
    ExecutionError,
    RiskManagementError,
    SignalGenerationError,
    ValidationError,
)
from src.core.types import OrderRequest, OrderSide, Signal, SignalDirection
from src.risk_management.service import RiskService


class RiskManagerAdapter:
    """
    Adapter to make RiskService compatible with execution algorithm expectations.

    Execution algorithms expect:
    - validate_order(order: OrderRequest, portfolio_value: Decimal) -> bool

    RiskService provides:
    - validate_order(order: OrderRequest) -> bool
    - validate_signal(signal: Signal) -> bool
    """

    def __init__(self, risk_service: RiskService):
        """
        Initialize adapter with RiskService instance.

        Args:
            risk_service: The RiskService instance to adapt
        """
        self.risk_service = risk_service
        self._logger = logging.getLogger(__name__)

    async def validate_order(self, order: OrderRequest, portfolio_value: Decimal) -> bool:
        """
        Validate order using RiskService.

        This method adapts the algorithm's expected interface to RiskService's actual interface.

        Args:
            order: OrderRequest object from algorithm
            portfolio_value: Portfolio value (not used by RiskService)

        Returns:
            bool: True if order is valid, False otherwise
        """
        try:
            # order is already OrderRequest type
            order_request = order

            # Use RiskService's validate_order method
            result = await self.risk_service.validate_order(order_request)
            return result

        except ValidationError as e:
            # If validation fails due to invalid inputs, try signal validation as fallback
            self._logger.warning(
                f"Order validation failed due to invalid inputs, trying signal validation: {e}"
            )
            try:
                # Map OrderSide to SignalDirection
                signal_direction = (
                    SignalDirection.BUY
                    if order.side == OrderSide.BUY
                    else (
                        SignalDirection.SELL
                        if order.side == OrderSide.SELL
                        else SignalDirection.HOLD
                    )
                )

                # Create Signal for validation
                signal = Signal(
                    symbol=order.symbol,
                    direction=signal_direction,
                    strength=0.5,  # Default confidence
                    timestamp=datetime.now(timezone.utc),
                    source="execution_algorithm",
                    metadata={
                        "quantity": str(order.quantity),
                        "price": str(order.price) if order.price else "0",
                        "order_type": order.order_type.value,
                        "portfolio_value": str(portfolio_value),
                    },
                )

                # Validate as signal
                result = await self.risk_service.validate_signal(signal)
                return result

            except (RiskManagementError, ValidationError, SignalGenerationError) as fallback_error:
                # Log the fallback failure and re-raise for safety
                self._logger.error(
                    f"Both order and signal validation failed for {order.symbol}: {fallback_error}"
                )
                raise RiskManagementError(
                    f"Risk validation failed for {order.symbol}: {fallback_error}"
                ) from fallback_error
        except (RiskManagementError, ExecutionError) as e:
            # For risk management errors that are not validation errors, re-raise
            self._logger.error(f"Risk management error during validation: {e}")
            raise
        except Exception as unexpected_error:
            # For truly unexpected errors, wrap and re-raise for safety
            self._logger.critical(
                f"Unexpected error during risk validation for {order.symbol}: {unexpected_error}"
            )
            raise ExecutionError(
                f"Unexpected error during risk validation: {unexpected_error}"
            ) from unexpected_error

    async def calculate_position_size(
        self,
        symbol: str,
        side: OrderSide,
        confidence: float,
        current_price: Decimal,
        available_capital: Decimal | None = None,
    ) -> Decimal:
        """
        Calculate position size using RiskService.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            confidence: Signal confidence (0-1)
            current_price: Current market price
            available_capital: Available capital (optional)

        Returns:
            Decimal: Calculated position size
        """
        # Map OrderSide to SignalDirection
        signal_direction = (
            SignalDirection.BUY
            if side == OrderSide.BUY
            else SignalDirection.SELL
            if side == OrderSide.SELL
            else SignalDirection.HOLD
        )

        # Create Signal for position sizing
        signal = Signal(
            symbol=symbol,
            direction=signal_direction,
            strength=confidence,
            timestamp=datetime.now(timezone.utc),
            source="execution_algorithm",
            metadata={},
        )

        # Calculate position size
        position_size = await self.risk_service.calculate_position_size(
            signal=signal,
            available_capital=available_capital,
            current_price=current_price,
        )

        return position_size or Decimal("0")

    # Pass through other RiskService methods that might be needed
    async def get_risk_summary(self) -> dict[str, Any]:
        """Get risk summary from RiskService."""
        return await self.risk_service.get_risk_summary()

    async def calculate_risk_metrics(self, positions: list, market_data: list) -> Any:
        """Calculate risk metrics using RiskService."""
        return await self.risk_service.calculate_risk_metrics(positions, market_data)
