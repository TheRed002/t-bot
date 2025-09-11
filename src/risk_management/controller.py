"""
Risk Management Controller.

This controller handles risk management operations by delegating to appropriate
services, following proper controller->service->repository patterns.
"""

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.core.base.component import BaseComponent
from src.core.types import MarketData, OrderRequest, Position, RiskMetrics, Signal
from src.utils.messaging_patterns import (
    BoundaryValidator,
    ErrorPropagationMixin,
    MessagingCoordinator,
)

if TYPE_CHECKING:
    from .interfaces import (
        PositionSizingServiceInterface,
        RiskMetricsServiceInterface,
        RiskMonitoringServiceInterface,
        RiskValidationServiceInterface,
    )


class RiskManagementController(BaseComponent, ErrorPropagationMixin):
    """
    Controller for risk management operations.

    This controller delegates all business logic to appropriate services,
    maintaining proper separation of concerns and consistent data flow patterns.
    """

    def __init__(
        self,
        position_sizing_service: "PositionSizingServiceInterface",
        risk_validation_service: "RiskValidationServiceInterface",
        risk_metrics_service: "RiskMetricsServiceInterface",
        risk_monitoring_service: "RiskMonitoringServiceInterface",
        messaging_coordinator: MessagingCoordinator | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize risk management controller.

        Args:
            position_sizing_service: Service for position sizing
            risk_validation_service: Service for risk validation
            risk_metrics_service: Service for risk metrics
            risk_monitoring_service: Service for risk monitoring
            messaging_coordinator: Messaging coordinator for consistent data flow
            correlation_id: Request correlation ID
        """
        super().__init__()
        self._request_count = 0
        self._position_sizing_service = position_sizing_service
        self._risk_validation_service = risk_validation_service
        self._risk_metrics_service = risk_metrics_service
        self._risk_monitoring_service = risk_monitoring_service
        self._messaging_coordinator = messaging_coordinator or MessagingCoordinator(
            "RiskManagementController"
        )

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
            # Validate input at boundary
            BoundaryValidator.validate_database_entity(
                {
                    "signal": signal.model_dump(),
                    "available_capital": str(available_capital),
                    "current_price": str(current_price),
                },
                "calculate_position_size",
            )

            self.logger.info(
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

            self.logger.info(
                "Position size calculated",
                symbol=signal.symbol,
                size=str(position_size),
                method=method,
            )

            return position_size

        except Exception as e:
            self.propagate_service_error(e, "calculate_position_size")
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
            # Validate input at boundary
            BoundaryValidator.validate_database_entity(
                {"signal": signal.model_dump()}, "validate_signal"
            )

            self.logger.info("Validating signal", symbol=signal.symbol)

            # Delegate to risk validation service
            is_valid = await self._risk_validation_service.validate_signal(signal)

            self.logger.info(
                "Signal validation result",
                symbol=signal.symbol,
                valid=is_valid,
            )

            return is_valid

        except Exception as e:
            self.propagate_validation_error(e, "validate_signal")
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
            # Validate input at boundary
            BoundaryValidator.validate_database_entity(
                {"order": order.model_dump()}, "validate_order"
            )

            self.logger.info("Validating order", symbol=order.symbol)

            # Delegate to risk validation service
            is_valid = await self._risk_validation_service.validate_order(order)

            self.logger.info(
                "Order validation result",
                symbol=order.symbol,
                valid=is_valid,
            )

            return is_valid

        except Exception as e:
            self.propagate_validation_error(e, "validate_order")
            return False

    async def calculate_risk_metrics(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> RiskMetrics:
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
            # Validate inputs at boundary
            positions_data = [pos.model_dump() for pos in positions]
            market_data_dict = [md.model_dump() for md in market_data]
            BoundaryValidator.validate_database_entity(
                {"positions": positions_data, "market_data": market_data_dict},
                "calculate_risk_metrics",
            )

            self.logger.info(
                "Calculating risk metrics",
                position_count=len(positions),
                market_data_count=len(market_data),
            )

            # Use stream processing pattern for consistency with execution module
            from src.risk_management.data_transformer import RiskDataTransformer

            # Transform positions data for stream processing consistency
            _ = [
                RiskDataTransformer.transform_position_to_event_data(
                    pos, {"correlation_id": self.correlation_id}
                )
                for pos in positions
            ]

            # Delegate to risk metrics service
            metrics = await self._risk_metrics_service.calculate_metrics(
                positions=positions,
                market_data=market_data,
            )

            self.logger.info(
                "Risk metrics calculated",
                portfolio_value=str(metrics.portfolio_value),
                risk_level=metrics.risk_level.value,
            )

            return metrics

        except Exception as e:
            self.propagate_service_error(e, "calculate_risk_metrics")
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
            self.logger.info(
                "Validating portfolio limits",
                symbol=new_position.symbol,
            )

            # Delegate to risk validation service
            is_valid = await self._risk_validation_service.validate_portfolio_limits(new_position)

            self.logger.info(
                "Portfolio limits validation result",
                symbol=new_position.symbol,
                valid=is_valid,
            )

            return is_valid

        except Exception as e:
            self.logger.error(f"Portfolio limits validation failed: {e}")
            return False

    async def start_monitoring(self, interval: int = 60) -> None:
        """
        Start risk monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        try:
            self.logger.info(f"Starting risk monitoring with {interval}s interval")

            # Delegate to risk monitoring service
            await self._risk_monitoring_service.start_monitoring(interval)

            self.logger.info("Risk monitoring started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start risk monitoring: {e}")
            raise

    async def stop_monitoring(self) -> None:
        """Stop risk monitoring."""
        try:
            self.logger.info("Stopping risk monitoring")

            # Delegate to risk monitoring service
            await self._risk_monitoring_service.stop_monitoring()

            self.logger.info("Risk monitoring stopped successfully")

        except Exception as e:
            self.logger.error(f"Failed to stop risk monitoring: {e}")
            raise

    async def get_risk_summary(self) -> dict[str, Any]:
        """
        Get comprehensive risk summary.

        Returns:
            Dictionary with risk summary information
        """
        try:
            self.logger.info("Getting risk summary")

            # Delegate all business logic to monitoring service
            summary = await self._risk_monitoring_service.get_risk_summary()

            self.logger.info("Risk summary retrieved from service")
            return summary

        except Exception as e:
            self.logger.error(f"Failed to get risk summary: {e}")
            from datetime import datetime, timezone

            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
