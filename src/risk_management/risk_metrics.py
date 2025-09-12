"""
Risk Metrics Calculator - Refactored to use centralized utilities.

This module now delegates to centralized risk calculation utilities,
eliminating code duplication while maintaining backward compatibility.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.base import BaseComponent
from src.core.config import Config
from src.core.exceptions import RiskManagementError
from src.core.types import MarketData, Position, RiskLevel, RiskMetrics
from src.utils.decorators import time_execution

# Import centralized risk calculation utilities
from src.utils.risk_calculations import (
    calculate_current_drawdown,
    calculate_expected_shortfall,
    calculate_max_drawdown,
    calculate_portfolio_value,
    calculate_sharpe_ratio,
    calculate_var,
    determine_risk_level,
    update_returns_history,
    validate_risk_inputs,
)

# No external dependencies required for this component


class RiskCalculator(BaseComponent):
    """
    Risk metrics calculator for portfolio risk assessment.

    This class now delegates to centralized utilities to eliminate code duplication.
    Use RiskService.calculate_risk_metrics() for new implementations.
    """

    def __init__(self, config: Config):
        """
        Initialize risk calculator with configuration.

        Args:
            config: Application configuration containing risk settings
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.risk_config = config.risk

        # Historical data for calculations - using centralized utilities
        self.portfolio_values: list[Decimal] = []
        self.portfolio_returns: list[Decimal] = []
        self.position_returns: dict[str, list[Decimal]] = {}
        self.position_prices: dict[str, list[Decimal]] = {}

        self.logger.info(
            "RiskCalculator initialized - consider using RiskMetricsService for enterprise features"
        )

    @time_execution
    async def calculate_risk_metrics(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics using centralized utilities.

        Args:
            positions: Current portfolio positions
            market_data: Current market data for all positions

        Returns:
            RiskMetrics: Calculated risk metrics

        Raises:
            RiskManagementError: If risk calculation fails
        """
        try:
            # Validate inputs using centralized utility
            if not positions:
                return await self._create_empty_risk_metrics()

            # Calculate portfolio value using centralized utility
            portfolio_value = calculate_portfolio_value(positions, market_data)

            # Validate inputs using centralized utility
            if not validate_risk_inputs(portfolio_value, positions, market_data):
                raise RiskManagementError(
                    "Risk input validation failed - invalid positions or market data"
                )

            # Update historical data
            await self._update_portfolio_history(portfolio_value)

            # Calculate risk metrics using centralized utilities
            var_1d = calculate_var(self.portfolio_returns, Decimal("0.95"), 1)
            var_5d = calculate_var(self.portfolio_returns, Decimal("0.95"), 5)
            expected_shortfall = calculate_expected_shortfall(self.portfolio_returns)
            max_drawdown, _, _ = calculate_max_drawdown(self.portfolio_values)
            current_drawdown = calculate_current_drawdown(
                portfolio_value, self.portfolio_values[:-1]
            )
            sharpe_ratio = calculate_sharpe_ratio(self.portfolio_returns)

            # Determine risk level using centralized utility
            risk_level = determine_risk_level(
                var_1d, current_drawdown, sharpe_ratio, portfolio_value
            )

            total_exposure = portfolio_value

            # Create risk metrics object
            risk_metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=total_exposure,
                var_1d=var_1d,
                var_5d=var_5d,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                current_drawdown=current_drawdown,
                risk_level=risk_level,
                timestamp=datetime.now(timezone.utc),
            )

            self.logger.info(
                "Risk metrics calculated",
                var_1d=str(var_1d),
                var_5d=str(var_5d),
                current_drawdown=str(current_drawdown),
                risk_level=risk_level.value,
            )

            return risk_metrics

        except Exception as e:
            self.logger.error("Risk metrics calculation failed", error=str(e))
            raise RiskManagementError(f"Risk metrics calculation failed: {e}") from e

    @time_execution
    async def _create_empty_risk_metrics(self) -> RiskMetrics:
        """
        Create empty risk metrics for portfolios with no positions.

        Returns:
            RiskMetrics: Empty risk metrics
        """
        return RiskMetrics(
            portfolio_value=Decimal("0"),
            total_exposure=Decimal("0"),
            var_1d=Decimal("0"),
            var_5d=Decimal("0"),
            expected_shortfall=Decimal("0"),
            max_drawdown=Decimal("0"),
            sharpe_ratio=None,
            current_drawdown=Decimal("0"),
            risk_level=RiskLevel.LOW,
            timestamp=datetime.now(timezone.utc),
        )

    @time_execution
    async def _update_portfolio_history(self, portfolio_value: Decimal) -> None:
        """
        Update portfolio value history for risk calculations.

        Args:
            portfolio_value: Current portfolio value
        """
        self.portfolio_values.append(portfolio_value)

        # Update returns using centralized utility
        self.portfolio_returns = update_returns_history(
            self.portfolio_values,
            max_history=max(getattr(self.risk_config, "correlation_window", 30), 252),
        )

        # Keep only recent history
        max_history = max(getattr(self.risk_config, "correlation_window", 30), 252)
        if len(self.portfolio_values) > max_history:
            self.portfolio_values = self.portfolio_values[-max_history:]

    @time_execution
    async def update_position_returns(self, symbol: str, price: Decimal) -> None:
        """
        Update position return history for individual position risk.

        Args:
            symbol: Trading symbol
            price: Current price
        """
        if symbol not in self.position_returns:
            self.position_returns[symbol] = []
            self.position_prices[symbol] = []

        # Store the current price as Decimal
        self.position_prices[symbol].append(price)

        # Update returns using centralized utility
        self.position_returns[symbol] = update_returns_history(self.position_prices[symbol])

        # Keep only recent history
        max_history = 252  # One year of trading days
        if len(self.position_returns[symbol]) > max_history:
            self.position_returns[symbol] = self.position_returns[symbol][-max_history:]
        if len(self.position_prices[symbol]) > max_history:
            self.position_prices[symbol] = self.position_prices[symbol][-max_history:]

    @time_execution
    async def get_risk_summary(self) -> dict[str, Any]:
        """
        Get comprehensive risk summary.

        Returns:
            Dict containing current risk state and metrics
        """
        if not self.portfolio_values:
            return {"error": "No portfolio data available"}

        current_value = self.portfolio_values[-1] if self.portfolio_values else Decimal("0")
        peak_value = max(self.portfolio_values) if self.portfolio_values else Decimal("0")

        summary = {
            "current_portfolio_value": str(current_value),
            "peak_portfolio_value": str(peak_value),
            "total_return": (
                str((current_value - self.portfolio_values[0]) / self.portfolio_values[0])
                if len(self.portfolio_values) > 1 and self.portfolio_values[0] > Decimal("0")
                else "0.0"
            ),
            "data_points": len(self.portfolio_values),
            "return_data_points": len(self.portfolio_returns),
            "position_symbols": list(self.position_returns.keys()),
        }

        return summary
