"""
Risk Manager Adapter for Execution Algorithms.

This adapter bridges the interface mismatch between what execution algorithms
expect (risk_manager.validate_order) and what RiskService provides.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.types import Order, OrderRequest, OrderSide, Signal, SignalDirection
from src.risk_management.service import RiskService


class RiskManagerAdapter:
    """
    Adapter to make RiskService compatible with execution algorithm expectations.

    Execution algorithms expect:
    - validate_order(order: Order, portfolio_value: Decimal) -> bool

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

    async def validate_order(self, order: Order, portfolio_value: Decimal) -> bool:
        """
        Validate order using RiskService.

        This method adapts the algorithm's expected interface to RiskService's actual interface.

        Args:
            order: Order object from algorithm
            portfolio_value: Portfolio value (not used by RiskService)

        Returns:
            bool: True if order is valid, False otherwise
        """
        try:
            # Create OrderRequest from Order
            order_request = OrderRequest(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price if hasattr(order, "stop_price") else None,
                time_in_force=order.time_in_force if hasattr(order, "time_in_force") else None,
                client_order_id=order.order_id,
                metadata={
                    "source": "execution_algorithm",
                    "portfolio_value": float(portfolio_value),
                },
            )

            # Use RiskService's validate_order method
            result = await self.risk_service.validate_order(order_request)
            return result

        except Exception:
            # If validation fails, we can also try signal validation as fallback
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
                        "quantity": float(order.quantity),
                        "price": float(order.price) if order.price else 0.0,
                        "order_type": order.order_type.value,
                        "portfolio_value": float(portfolio_value),
                    },
                )

                # Validate as signal
                result = await self.risk_service.validate_signal(signal)
                return result

            except:
                # If all else fails, return False (reject order)
                return False

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
