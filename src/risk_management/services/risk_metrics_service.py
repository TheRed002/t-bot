"""
Risk Metrics Service Implementation.

This service handles all risk metrics calculations through dependency injection,
following proper service layer patterns.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

from src.core.base.service import BaseService
from src.core.exceptions import RiskManagementError
from src.core.types import MarketData, Position, RiskLevel, RiskMetrics
from src.utils.decimal_utils import ONE, ZERO, format_decimal, safe_divide, to_decimal

if TYPE_CHECKING:
    from src.database.service import DatabaseService
    from src.state import StateService


class RiskMetricsService(BaseService):
    """Service for calculating comprehensive risk metrics."""

    def __init__(
        self,
        database_service: "DatabaseService",
        state_service: "StateService",
        config=None,
        correlation_id: str | None = None,
    ):
        """
        Initialize risk metrics service.

        Args:
            database_service: Database service for data access
            state_service: State service for state management
            config: Application configuration
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="RiskMetricsService",
            config=config.__dict__ if config else {},
            correlation_id=correlation_id,
        )

        self.database_service = database_service
        self.state_service = state_service
        self.config = config

    async def calculate_metrics(self, positions: list[Position], market_data: list[MarketData]) -> RiskMetrics:
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
            if not positions:
                return await self._create_empty_metrics()

            # Calculate portfolio value
            portfolio_value = await self.get_portfolio_value(positions, market_data)

            # Update portfolio history
            await self._update_portfolio_history(portfolio_value)

            # Get historical data for calculations
            portfolio_history = await self._get_portfolio_history()

            # Calculate individual risk components
            var_1d = await self._calculate_var(1, portfolio_value, portfolio_history)
            var_5d = await self._calculate_var(5, portfolio_value, portfolio_history)
            expected_shortfall = await self._calculate_expected_shortfall(portfolio_value, portfolio_history)
            max_drawdown = await self._calculate_max_drawdown(portfolio_history)
            current_drawdown = await self._calculate_current_drawdown(portfolio_value, portfolio_history)
            sharpe_ratio = await self._calculate_sharpe_ratio(portfolio_history)

            # Calculate additional metrics
            total_exposure = await self._calculate_total_exposure(positions, market_data)
            correlation_risk = await self._calculate_correlation_risk(positions)
            beta = await self._calculate_portfolio_beta(positions)

            # Determine risk level
            risk_level = await self._determine_risk_level(var_1d, current_drawdown, sharpe_ratio, portfolio_value)

            # Create risk metrics
            metrics = RiskMetrics(
                timestamp=datetime.now(timezone.utc),
                portfolio_value=portfolio_value,
                total_exposure=total_exposure,
                var_1d=var_1d,
                var_95=var_1d,  # Same as var_1d for 95% confidence
                var_99=var_5d,  # Approximation for 99% confidence
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=float(sharpe_ratio) if sharpe_ratio else None,
                beta=float(beta) if beta else None,
                correlation_risk=correlation_risk,
                risk_level=risk_level,
                position_count=len(positions),
                leverage=float(safe_divide(total_exposure, portfolio_value, ONE)),
            )

            # Store metrics in state
            await self._store_metrics(metrics)

            self._logger.info(
                "Risk metrics calculated",
                portfolio_value=format_decimal(portfolio_value),
                var_1d=format_decimal(var_1d),
                risk_level=risk_level.value,
            )

            return metrics

        except Exception as e:
            self._logger.error(f"Risk metrics calculation failed: {e}")
            raise RiskManagementError(f"Risk metrics calculation failed: {e}") from e

    async def get_portfolio_value(self, positions: list[Position], market_data: list[MarketData]) -> Decimal:
        """
        Calculate current portfolio value.

        Args:
            positions: Current positions
            market_data: Market data for positions

        Returns:
            Current portfolio value
        """
        portfolio_value = ZERO

        # Create price lookup
        price_lookup = {data.symbol: data.close for data in market_data}

        for position in positions:
            current_price = price_lookup.get(position.symbol, position.current_price)
            if current_price and current_price > ZERO:
                position_value = position.quantity * current_price
                portfolio_value += position_value

        return portfolio_value

    async def _create_empty_metrics(self) -> RiskMetrics:
        """Create empty risk metrics for portfolios with no positions."""
        return RiskMetrics(
            timestamp=datetime.now(timezone.utc),
            portfolio_value=ZERO,
            total_exposure=ZERO,
            var_1d=ZERO,
            var_95=ZERO,
            var_99=ZERO,
            expected_shortfall=ZERO,
            max_drawdown=ZERO,
            current_drawdown=ZERO,
            sharpe_ratio=None,
            beta=None,
            correlation_risk=ZERO,
            risk_level=RiskLevel.LOW,
            position_count=0,
            leverage=ONE,
        )

    async def _update_portfolio_history(self, portfolio_value: Decimal) -> None:
        """Update portfolio value history."""
        try:
            # Store portfolio value in state service
            history_data = await self.state_service.get_state("risk", "portfolio_history") or []
            history_data.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "value": str(portfolio_value),
                }
            )

            if len(history_data) > 252:
                history_data = history_data[-252:]

            await self.state_service.set_state("risk", "portfolio_history", history_data)

        except Exception as e:
            self._logger.error(f"Error updating portfolio history: {e}")

    async def _get_portfolio_history(self) -> list[Decimal]:
        """Get portfolio value history."""
        try:
            history_data = await self.state_service.get_state("risk", "portfolio_history") or []
            return [to_decimal(item["value"]) for item in history_data if "value" in item]
        except Exception as e:
            self._logger.error(f"Error getting portfolio history: {e}")
            return []

    async def _calculate_var(self, days: int, portfolio_value: Decimal, history: list[Decimal]) -> Decimal:
        """Calculate Value at Risk."""
        if len(history) < 30:
            # Conservative estimate for insufficient data
            base_var_pct = to_decimal("0.02")  # 2% base VaR
            scaled_var_pct = base_var_pct * to_decimal(str(np.sqrt(days)))
            return portfolio_value * scaled_var_pct

        # Calculate returns using Decimal precision
        returns = []
        for i in range(1, len(history)):
            if history[i - 1] > ZERO:
                return_val = safe_divide(history[i] - history[i - 1], history[i - 1], ZERO)
                returns.append(float(return_val))  # Convert to float only for numpy operations

        if len(returns) < 10:
            return portfolio_value * to_decimal("0.02")

        # Calculate VaR using percentile method with proper precision
        returns_array = np.array(returns)
        daily_var = np.percentile(returns_array, 5)  # 95% confidence level

        # Scale for time horizon using Decimal for precision
        scaling_factor = Decimal(str(np.sqrt(days)))
        scaled_var = to_decimal(str(daily_var)) * scaling_factor

        return portfolio_value * abs(scaled_var)

    async def _calculate_expected_shortfall(self, portfolio_value: Decimal, history: list[Decimal]) -> Decimal:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(history) < 30:
            return portfolio_value * to_decimal("0.025")

        # Calculate returns using Decimal precision
        returns = []
        for i in range(1, len(history)):
            if history[i - 1] > ZERO:
                return_val = (history[i] - history[i - 1]) / history[i - 1]
                returns.append(float(return_val))  # Convert to float only for numpy operations

        if not returns:
            return portfolio_value * to_decimal("0.025")

        # Find worst 5% of returns
        returns_array = np.array(returns)
        threshold = np.percentile(returns_array, 5)
        worst_returns = returns_array[returns_array <= threshold]

        if len(worst_returns) == 0:
            return portfolio_value * to_decimal("0.02")

        # Use Decimal for final calculation
        mean_worst = to_decimal(str(abs(np.mean(worst_returns))))
        expected_shortfall = portfolio_value * mean_worst
        return expected_shortfall

    async def _calculate_max_drawdown(self, history: list[Decimal]) -> Decimal:
        """Calculate maximum historical drawdown."""
        if len(history) < 2:
            return ZERO

        values = [float(val) for val in history]
        running_max = np.maximum.accumulate(values)
        drawdowns = (running_max - values) / running_max

        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        return to_decimal(str(max_drawdown))

    async def _calculate_current_drawdown(self, portfolio_value: Decimal, history: list[Decimal]) -> Decimal:
        """Calculate current drawdown from peak."""
        if len(history) < 2:
            return ZERO

        peak_value = max(history)
        if peak_value <= ZERO:
            return ZERO

        current_drawdown = safe_divide(peak_value - portfolio_value, peak_value, ZERO)
        return max(ZERO, current_drawdown)

    async def _calculate_sharpe_ratio(self, history: list[Decimal]) -> Decimal | None:
        """Calculate Sharpe ratio."""
        if len(history) < 30:
            return None

        # Calculate returns using Decimal precision
        returns = []
        for i in range(1, len(history)):
            if history[i - 1] > ZERO:
                return_val = (history[i] - history[i - 1]) / history[i - 1]
                returns.append(float(return_val))  # Convert to float only for numpy operations

        if len(returns) < 30:
            return None

        returns_array = np.array(returns)

        # Annualized metrics with Decimal precision
        mean_return = to_decimal(str(np.mean(returns_array) * 252))
        volatility = to_decimal(str(np.std(returns_array) * np.sqrt(252)))

        if volatility == ZERO:
            return None

        sharpe_ratio = safe_divide(mean_return, volatility, ZERO)
        return sharpe_ratio

    async def _calculate_total_exposure(self, positions: list[Position], market_data: list[MarketData]) -> Decimal:
        """Calculate total portfolio exposure."""
        total_exposure = ZERO
        price_lookup = {data.symbol: data.close for data in market_data}

        for position in positions:
            current_price = price_lookup.get(position.symbol, position.current_price)
            if current_price and current_price > ZERO:
                position_exposure = abs(position.quantity) * current_price
                total_exposure += position_exposure

        return total_exposure

    async def _calculate_correlation_risk(self, positions: list[Position]) -> Decimal:
        """Calculate correlation risk between positions."""
        if len(positions) < 2:
            return ZERO

        # Simplified correlation risk calculation using Decimal precision
        # In production, this would use actual correlation matrices
        unique_symbols = len(set(pos.symbol for pos in positions))
        if unique_symbols == 0:
            return ZERO

        concentration = safe_divide(ONE, to_decimal(str(unique_symbols)), ZERO)
        return ONE - concentration

    async def _calculate_portfolio_beta(self, positions: list[Position]) -> Decimal | None:
        """Calculate portfolio beta (placeholder for future implementation)."""
        # This would require market index data
        return None

    async def _determine_risk_level(
        self,
        var_1d: Decimal,
        current_drawdown: Decimal,
        sharpe_ratio: Decimal | None,
        portfolio_value: Decimal,
    ) -> RiskLevel:
        """Determine risk level based on metrics."""
        # Calculate VaR as percentage of portfolio
        var_1d_pct = safe_divide(var_1d, portfolio_value, ZERO) if portfolio_value > ZERO else ZERO

        # Risk thresholds
        var_critical = to_decimal("0.10")  # 10% VaR
        var_high = to_decimal("0.05")  # 5% VaR
        var_medium = to_decimal("0.02")  # 2% VaR

        drawdown_critical = to_decimal("0.20")  # 20% drawdown
        drawdown_high = to_decimal("0.10")  # 10% drawdown
        drawdown_medium = to_decimal("0.05")  # 5% drawdown

        # Check for critical risk
        if var_1d_pct > var_critical or current_drawdown > drawdown_critical:
            return RiskLevel.CRITICAL

        # Check for high risk
        if (
            var_1d_pct > var_high
            or current_drawdown > drawdown_high
            or (sharpe_ratio is not None and sharpe_ratio < to_decimal("-1.0"))
        ):
            return RiskLevel.HIGH

        # Check for medium risk
        if (
            var_1d_pct > var_medium
            or current_drawdown > drawdown_medium
            or (sharpe_ratio is not None and sharpe_ratio < to_decimal("0.5"))
        ):
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    async def _store_metrics(self, metrics: RiskMetrics) -> None:
        """Store calculated metrics in state service."""
        try:
            metrics_data = {
                "timestamp": metrics.timestamp.isoformat(),
                "portfolio_value": str(metrics.portfolio_value),
                "total_exposure": str(metrics.total_exposure),
                "var_1d": str(metrics.var_1d),
                "current_drawdown": str(metrics.current_drawdown),
                "risk_level": metrics.risk_level.value,
                "position_count": metrics.position_count,
            }

            await self.state_service.set_state("risk", "latest_metrics", metrics_data)

        except Exception as e:
            self._logger.error(f"Error storing risk metrics: {e}")
