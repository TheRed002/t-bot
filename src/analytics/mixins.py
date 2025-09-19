"""
Analytics Mixins - Common functionality for analytics services.

This module provides mixins to eliminate code duplication across analytics services.
"""

from typing import TYPE_CHECKING, Any

from src.core.exceptions import ComponentError
from src.core.types import Order, Position, Trade

if TYPE_CHECKING:
    pass


class PositionTrackingMixin:
    """Mixin for services that track positions and trades."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._positions: dict[str, Position] = {}
        self._trades: list[Trade] = []
        self._max_trades_history = kwargs.get("max_trades_history", 1000)

    def update_position(self, position: Position) -> None:
        """Update position data with standardized error handling."""
        try:
            self._positions[position.symbol] = position
            self.logger.debug(f"Updated position for {position.symbol}")
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
            raise ComponentError(
                f"Failed to update position {position.symbol}",
                error_code="ANL_017",
                component=self.__class__.__name__,
                operation="update_position",
            ) from e

    def update_trade(self, trade: Trade) -> None:
        """Update trade data with standardized error handling."""
        try:
            self._trades.append(trade)
            # Keep only last N trades to prevent memory bloat
            if len(self._trades) > self._max_trades_history:
                self._trades = self._trades[-self._max_trades_history :]
            self.logger.debug(f"Updated trade {trade.trade_id}")
        except Exception as e:
            self.logger.error(f"Error updating trade: {e}")
            raise ComponentError(
                f"Failed to update trade {trade.trade_id}",
                error_code="ANL_018",
                component=self.__class__.__name__,
                operation="update_trade",
            ) from e

    def get_position(self, symbol: str) -> Position | None:
        """Get position by symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        """Get all positions."""
        return self._positions.copy()

    def get_recent_trades(self, limit: int | None = None) -> list[Trade]:
        """Get recent trades, optionally limited."""
        if limit is None:
            return self._trades.copy()
        return self._trades[-limit:] if limit > 0 else []


class OrderTrackingMixin:
    """Mixin for services that track orders."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orders: dict[str, Order] = {}

    def update_order(self, order: Order) -> None:
        """Update order data with standardized error handling."""
        try:
            self._orders[order.order_id] = order
            self.logger.debug(f"Updated order {order.order_id}")
        except Exception as e:
            self.logger.error(f"Error updating order: {e}")
            raise ComponentError(
                f"Failed to update order {order.order_id}",
                error_code="ANL_019",
                component=self.__class__.__name__,
                operation="update_order",
            ) from e

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_all_orders(self) -> dict[str, Order]:
        """Get all orders."""
        return self._orders.copy()


# Use standardized error handling from core and error_handling modules
# Instead of creating redundant mixins, use existing infrastructure:
# - Error decorators from src.error_handling.decorators (with_retry, with_circuit_breaker, etc.)
# - Error handling service from src.error_handling.service
# - ErrorPropagationMixin from src.utils.messaging_patterns
from src.utils.messaging_patterns import ErrorPropagationMixin
