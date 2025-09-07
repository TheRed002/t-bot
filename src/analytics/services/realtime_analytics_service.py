"""
Realtime Analytics Service.

This service provides a proper service layer implementation for realtime analytics,
following service layer patterns and using dependency injection.
"""

from decimal import Decimal
from typing import Any

from src.analytics.interfaces import RealtimeAnalyticsServiceProtocol
from src.analytics.trading.realtime_analytics import RealtimeAnalyticsEngine
from src.analytics.types import (
    AnalyticsConfiguration,
    PortfolioMetrics,
    PositionMetrics,
    StrategyMetrics,
)
from src.core.base.service import BaseService
from src.core.exceptions import ComponentError, ValidationError
from src.core.types import Order, Position, Trade


class RealtimeAnalyticsService(BaseService, RealtimeAnalyticsServiceProtocol):
    """
    Service layer implementation for realtime analytics.

    This service acts as a facade over the RealtimeAnalyticsEngine,
    providing proper service layer abstraction and dependency injection.
    """

    def __init__(
        self,
        config: AnalyticsConfiguration,
        analytics_engine: RealtimeAnalyticsEngine | None = None,
        metrics_collector=None,
    ):
        """
        Initialize the realtime analytics service.

        Args:
            config: Analytics configuration
            analytics_engine: Injected analytics engine (optional)
            metrics_collector: Injected metrics collector (optional)
        """
        super().__init__()
        self.config = config

        # Use dependency injection - engine must be injected
        if analytics_engine is None:
            raise ComponentError(
                "analytics_engine must be injected via dependency injection",
                component="RealtimeAnalyticsService",
                operation="__init__",
                context={"missing_dependency": "analytics_engine"},
            )

        self._engine = analytics_engine

        if metrics_collector:
            self._engine.metrics_collector = metrics_collector

        self.logger.info("RealtimeAnalyticsService initialized")

    async def start(self) -> None:
        """Start the realtime analytics service."""
        try:
            await self._engine.start()
            self.logger.info("Realtime analytics service started")
        except Exception as e:
            raise ComponentError(
                f"Failed to start realtime analytics service: {e}",
                component="RealtimeAnalyticsService",
                operation="start",
            ) from e

    async def stop(self) -> None:
        """Stop the realtime analytics service."""
        try:
            await self._engine.stop()
            self.logger.info("Realtime analytics service stopped")
        except Exception as e:
            self.logger.error(f"Error stopping realtime analytics service: {e}")

    def update_position(self, position: Position) -> None:
        """
        Update position data.

        Args:
            position: Position to update

        Raises:
            ValidationError: If position data is invalid
            ComponentError: If update fails
        """
        if not isinstance(position, Position):
            raise ValidationError(
                "Invalid position parameter",
                field_name="position",
                field_value=type(position),
                expected_type="Position",
            )

        try:
            self._engine.update_position(position)
            self.logger.debug(f"Position updated: {position.symbol}")
        except Exception as e:
            raise ComponentError(
                f"Failed to update position: {e}",
                component="RealtimeAnalyticsService",
                operation="update_position",
                context={"symbol": position.symbol},
            ) from e

    def update_trade(self, trade: Trade) -> None:
        """
        Update trade data.

        Args:
            trade: Trade to update

        Raises:
            ValidationError: If trade data is invalid
            ComponentError: If update fails
        """
        if not isinstance(trade, Trade):
            raise ValidationError(
                "Invalid trade parameter",
                field_name="trade",
                field_value=type(trade),
                expected_type="Trade",
            )

        try:
            self._engine.update_trade(trade)
            self.logger.debug(f"Trade updated: {trade.trade_id}")
        except Exception as e:
            raise ComponentError(
                f"Failed to update trade: {e}",
                component="RealtimeAnalyticsService",
                operation="update_trade",
                context={"trade_id": trade.trade_id},
            ) from e

    def update_order(self, order: Order) -> None:
        """
        Update order data.

        Args:
            order: Order to update

        Raises:
            ValidationError: If order data is invalid
            ComponentError: If update fails
        """
        if not isinstance(order, Order):
            raise ValidationError(
                "Invalid order parameter",
                field_name="order",
                field_value=type(order),
                expected_type="Order",
            )

        try:
            self._engine.update_order(order)
            self.logger.debug(f"Order updated: {order.order_id}")
        except Exception as e:
            raise ComponentError(
                f"Failed to update order: {e}",
                component="RealtimeAnalyticsService",
                operation="update_order",
                context={"order_id": order.order_id},
            ) from e

    def update_price(self, symbol: str, price: Decimal) -> None:
        """
        Update price data.

        Args:
            symbol: Trading symbol
            price: Current price

        Raises:
            ValidationError: If parameters are invalid
            ComponentError: If update fails
        """
        if not isinstance(symbol, str) or not symbol:
            raise ValidationError(
                "Invalid symbol parameter",
                field_name="symbol",
                field_value=symbol,
                expected_type="non-empty str",
            )

        if not isinstance(price, Decimal) or price <= 0:
            raise ValidationError(
                "Invalid price parameter",
                field_name="price",
                field_value=price,
                validation_rule="positive Decimal",
            )

        try:
            self._engine.update_price(symbol, price)
            self.logger.debug(f"Price updated: {symbol} = {price}")
        except Exception as e:
            raise ComponentError(
                f"Failed to update price: {e}",
                component="RealtimeAnalyticsService",
                operation="update_price",
                context={"symbol": symbol, "price": str(price)},
            ) from e

    async def get_portfolio_metrics(self) -> PortfolioMetrics | None:
        """
        Get current portfolio metrics.

        Returns:
            Portfolio metrics or None if not available

        Raises:
            ComponentError: If retrieval fails
        """
        try:
            return await self._engine.get_portfolio_metrics()
        except Exception as e:
            raise ComponentError(
                f"Failed to get portfolio metrics: {e}",
                component="RealtimeAnalyticsService",
                operation="get_portfolio_metrics",
            ) from e

    async def get_position_metrics(self, symbol: str = None) -> list[PositionMetrics]:
        """
        Get position metrics.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of position metrics

        Raises:
            ComponentError: If retrieval fails
        """
        try:
            return await self._engine.get_position_metrics(symbol)
        except Exception as e:
            raise ComponentError(
                f"Failed to get position metrics: {e}",
                component="RealtimeAnalyticsService",
                operation="get_position_metrics",
                context={"symbol": symbol},
            ) from e

    async def get_strategy_metrics(self, strategy: str = None) -> list[StrategyMetrics]:
        """
        Get strategy performance metrics.

        Args:
            strategy: Optional strategy filter

        Returns:
            List of strategy metrics

        Raises:
            ComponentError: If retrieval fails
        """
        try:
            return await self._engine.get_strategy_metrics(strategy)
        except Exception as e:
            raise ComponentError(
                f"Failed to get strategy metrics: {e}",
                component="RealtimeAnalyticsService",
                operation="get_strategy_metrics",
                context={"strategy": strategy},
            ) from e

    async def get_active_alerts(self) -> list[Any]:
        """
        Get active alerts from the analytics engine.

        Returns:
            List of active alerts

        Raises:
            ComponentError: If retrieval fails
        """
        try:
            return await self._engine.get_active_alerts()
        except Exception as e:
            raise ComponentError(
                f"Failed to get active alerts: {e}",
                component="RealtimeAnalyticsService",
                operation="get_active_alerts",
            ) from e

    def get_service_status(self) -> dict[str, Any]:
        """
        Get service status information.

        Returns:
            Service status dictionary
        """
        return {
            "service_name": "RealtimeAnalyticsService",
            "engine_type": type(self._engine).__name__,
            "running": getattr(self._engine, "_running", False),
            "configuration": {
                "calculation_frequency": self.config.calculation_frequency.value,
                "risk_free_rate": str(self.config.risk_free_rate),
            },
        }
