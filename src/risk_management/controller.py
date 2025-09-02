"""
Risk Management Controller.

This controller handles risk management operations by delegating to appropriate
services, following proper controller->service->repository patterns.
"""

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.core.base.component import BaseComponent
from src.core.types import MarketData, OrderRequest, Position, RiskMetrics, Signal

if TYPE_CHECKING:
    from .interfaces import (
        PositionSizingServiceInterface,
        RiskMetricsServiceInterface,
        RiskMonitoringServiceInterface,
        RiskValidationServiceInterface,
    )


class RiskManagementController(BaseComponent):
    """
    Controller for risk management operations.

    This controller delegates all business logic to appropriate services,
    maintaining proper separation of concerns.
    """

    def __init__(
        self,
        position_sizing_service: "PositionSizingServiceInterface",
        risk_validation_service: "RiskValidationServiceInterface",
        risk_metrics_service: "RiskMetricsServiceInterface",
        risk_monitoring_service: "RiskMonitoringServiceInterface",
        correlation_id: str | None = None,
    ):
        """
        Initialize risk management controller.

        Args:
            position_sizing_service: Service for position sizing
            risk_validation_service: Service for risk validation
            risk_metrics_service: Service for risk metrics
            risk_monitoring_service: Service for risk monitoring
            correlation_id: Request correlation ID
        """
        super().__init__()
        self._request_count = 0
        self._position_sizing_service = position_sizing_service
        self._risk_validation_service = risk_validation_service
        self._risk_metrics_service = risk_metrics_service
        self._risk_monitoring_service = risk_monitoring_service

    async def calculate_position_size(
        self,
        signal: Signal,
        available_capital: Decimal,
        current_price: Decimal,
        method: str | None = None,
    ) -> Decimal:
        """
        Calculate optimal position size.

        Args:
            signal: Trading signal
            available_capital: Available capital
            current_price: Current market price
            method: Position sizing method

        Returns:
            Calculated position size

        Raises:
            RiskManagementError: If calculation fails
            ValidationError: If inputs are invalid
        """
        try:
            self._logger.info(
                "Calculating position size",
                symbol=signal.symbol,
                available_capital=str(available_capital),
                method=method,
            )

            # Delegate to position sizing service
            position_size = await self._position_sizing_service.calculate_size(
                signal=signal,
                available_capital=available_capital,
                current_price=current_price,
                method=method,
            )

            self._logger.info(
                "Position size calculated",
                symbol=signal.symbol,
                size=str(position_size),
                method=method,
            )

            return position_size

        except Exception as e:
            self._logger.error(f"Position size calculation failed: {e}")
            raise

    async def validate_signal(self, signal: Signal) -> bool:
        """
        Validate trading signal against risk constraints.

        Args:
            signal: Trading signal to validate

        Returns:
            True if signal passes validation
        """
        try:
            self._logger.info("Validating signal", symbol=signal.symbol)

            # Delegate to risk validation service
            is_valid = await self._risk_validation_service.validate_signal(signal)

            self._logger.info(
                "Signal validation result",
                symbol=signal.symbol,
                valid=is_valid,
            )

            return is_valid

        except Exception as e:
            self._logger.error(f"Signal validation failed: {e}")
            return False

    async def validate_order(self, order: OrderRequest) -> bool:
        """
        Validate order against risk constraints.

        Args:
            order: Order request to validate

        Returns:
            True if order passes validation
        """
        try:
            self._logger.info("Validating order", symbol=order.symbol)

            # Delegate to risk validation service
            is_valid = await self._risk_validation_service.validate_order(order)

            self._logger.info(
                "Order validation result",
                symbol=order.symbol,
                valid=is_valid,
            )

            return is_valid

        except Exception as e:
            self._logger.error(f"Order validation failed: {e}")
            return False

    async def calculate_risk_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.

        Args:
            positions: Current portfolio positions
            market_data: Current market data

        Returns:
            Calculated risk metrics

        Raises:
            RiskManagementError: If calculation fails
        """
        try:
            self._logger.info(
                "Calculating risk metrics",
                position_count=len(positions),
                market_data_count=len(market_data),
            )

            # Delegate to risk metrics service
            metrics = await self._risk_metrics_service.calculate_metrics(
                positions=positions,
                market_data=market_data,
            )

            self._logger.info(
                "Risk metrics calculated",
                portfolio_value=str(metrics.portfolio_value),
                risk_level=metrics.risk_level.value,
            )

            return metrics

        except Exception as e:
            self._logger.error(f"Risk metrics calculation failed: {e}")
            raise

    async def validate_portfolio_limits(self, new_position: Position) -> bool:
        """
        Validate that adding a position won't violate portfolio limits.

        Args:
            new_position: Position to be added

        Returns:
            True if position addition is allowed
        """
        try:
            self._logger.info(
                "Validating portfolio limits",
                symbol=new_position.symbol,
            )

            # Delegate to risk validation service
            is_valid = await self._risk_validation_service.validate_portfolio_limits(new_position)

            self._logger.info(
                "Portfolio limits validation result",
                symbol=new_position.symbol,
                valid=is_valid,
            )

            return is_valid

        except Exception as e:
            self._logger.error(f"Portfolio limits validation failed: {e}")
            return False

    async def start_monitoring(self, interval: int = 60) -> None:
        """
        Start risk monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        try:
            self._logger.info(f"Starting risk monitoring with {interval}s interval")

            # Delegate to risk monitoring service
            await self._risk_monitoring_service.start_monitoring(interval)

            self._logger.info("Risk monitoring started successfully")

        except Exception as e:
            self._logger.error(f"Failed to start risk monitoring: {e}")
            raise

    async def stop_monitoring(self) -> None:
        """Stop risk monitoring."""
        try:
            self._logger.info("Stopping risk monitoring")

            # Delegate to risk monitoring service
            await self._risk_monitoring_service.stop_monitoring()

            self._logger.info("Risk monitoring stopped successfully")

        except Exception as e:
            self._logger.error(f"Failed to stop risk monitoring: {e}")
            raise

    async def get_risk_summary(self) -> dict[str, Any]:
        """
        Get comprehensive risk summary.

        Returns:
            Dictionary with risk summary information
        """
        try:
            self._logger.info("Getting risk summary")

            # Note: get_active_alerts is not part of the interface
            # alerts = await self._risk_monitoring_service.get_active_alerts(limit=10)

            summary = {
                "active_alerts_count": 0,  # Placeholder until interface is extended
                "monitoring_active": True,
                "timestamp": self._get_current_timestamp().isoformat(),
                "controller_metrics": {
                    "requests_processed": self._request_count,
                    "correlation_id": self.correlation_id,
                },
            }

            alert_count = summary.get("active_alerts_count", 0)
            self._logger.info("Risk summary generated", alert_count=alert_count)

            return summary

        except Exception as e:
            self._logger.error(f"Failed to get risk summary: {e}")
            return {"error": str(e), "timestamp": self._get_current_timestamp().isoformat()}

    def _get_current_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc)
