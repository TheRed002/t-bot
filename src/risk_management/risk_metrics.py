"""
Risk Metrics Calculator - DEPRECATED.

DEPRECATED: This module is deprecated in favor of RiskService.
The new RiskService (src/risk_management/service.py) provides all risk metrics
functionality with:
- DatabaseService integration (no direct DB access)
- StateService integration for state management
- Comprehensive caching layer with Redis
- Enhanced error handling with circuit breakers
- Real-time risk monitoring and alerts

This module is maintained for backward compatibility only.
New implementations should use RiskService.calculate_risk_metrics() directly.

Legacy risk metrics (now in RiskService):
- Value at Risk (VaR) with multiple time horizons
- Expected Shortfall (Conditional VaR)
- Maximum Drawdown with historical analysis
- Sharpe Ratio with proper annualization
- Current Drawdown from peak
- Risk Level Assessment with dynamic thresholds
- Portfolio Beta and Correlation Risk

CRITICAL: This integrates with P-001 (types, exceptions, config),
P-002A (error handling), and P-007A (utils) components.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.base.component import BaseComponent
from src.error_handling.context import ErrorContext
from src.core.config.main import Config
from src.core.exceptions import RiskManagementError, ValidationError

# MANDATORY: Import from P-001
from src.core.types import MarketData, Position, RiskLevel, RiskMetrics

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from src.database.service import DatabaseService

# MANDATORY: Import from P-002A
from src.error_handling import ErrorHandler

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution


class RiskCalculator(BaseComponent):
    """
    DEPRECATED: Risk metrics calculator for portfolio risk assessment.

    This class is deprecated in favor of RiskService which provides:
    - Enterprise-grade service architecture
    - DatabaseService integration (no direct DB access)
    - Enhanced caching with Redis
    - Real-time monitoring and alerting
    - State management integration

    This class calculates comprehensive risk metrics to assess
    portfolio risk and determine appropriate risk levels.

    DEPRECATED METHODS -> USE RiskService INSTEAD:
    - calculate_risk_metrics() -> RiskService.calculate_risk_metrics()
    - _calculate_var() -> Built into RiskService with caching
    - _calculate_expected_shortfall() -> Built into RiskService
    - _calculate_max_drawdown() -> Built into RiskService with history
    - _calculate_sharpe_ratio() -> Built into RiskService
    - _determine_risk_level() -> Built into RiskService with alerts
    """

    def __init__(self, config: Config, database_service: "DatabaseService | None" = None):
        """
        Initialize DEPRECATED risk calculator with configuration.

        DEPRECATED: Use RiskService.calculate_risk_metrics() directly.

        Args:
            config: Application configuration containing risk settings
            database_service: Database service for data access (not used in legacy mode)
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.risk_config = config.risk
        self.error_handler = ErrorHandler(config)
        # Note: logger is a property from BaseComponent, no need to bind

        # Database service integration (optional for backward compatibility)
        self.database_service = database_service

        # DEPRECATED: Historical data for calculations
        # NOTE: RiskService handles this with proper caching and state management
        self.portfolio_values: list[float] = []
        self.portfolio_returns: list[float] = []
        self.position_returns: dict[str, list[float]] = {}
        self.position_prices: dict[str, list[float]] = {}

        if database_service:
            self.logger.warning(
                "RiskCalculator initialized with DatabaseService - "
                "consider migrating to RiskService for full integration"
            )
        else:
            self.logger.warning(
                "DEPRECATED RiskCalculator initialized in legacy mode - "
                "migrate to RiskService for enterprise features"
            )

    @time_execution
    async def calculate_risk_metrics(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for the portfolio.

        Args:
            positions: Current portfolio positions
            market_data: Current market data for all positions

        Returns:
            RiskMetrics: Calculated risk metrics

        Raises:
            RiskManagementError: If risk calculation fails
        """
        try:
            # Validate inputs
            if not positions:
                return await self._create_empty_risk_metrics()

            if len(positions) != len(market_data):
                raise ValidationError("Position and market data count mismatch")

            # Calculate portfolio value and returns
            portfolio_value = await self._calculate_portfolio_value(positions, market_data)
            await self._update_portfolio_history(portfolio_value)

            # Calculate risk metrics
            var_1d = await self._calculate_var(1, portfolio_value)
            var_5d = await self._calculate_var(5, portfolio_value)
            expected_shortfall = await self._calculate_expected_shortfall(portfolio_value)
            max_drawdown = await self._calculate_max_drawdown()
            current_drawdown = await self._calculate_current_drawdown(portfolio_value)
            sharpe_ratio = await self._calculate_sharpe_ratio()

            # Determine risk level
            risk_level = await self._determine_risk_level(var_1d, current_drawdown, sharpe_ratio)

            # Calculate total exposure (same as portfolio value for spot trading)
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
                var_1d=float(var_1d),
                var_5d=float(var_5d),
                current_drawdown=float(current_drawdown),
                risk_level=risk_level.value,
            )

            return risk_metrics

        except Exception as e:
            self.logger.error("Risk metrics calculation failed", error=str(e))
            raise RiskManagementError(f"Risk metrics calculation failed: {e}")

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
    async def _calculate_portfolio_value(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> Decimal:
        """
        Calculate current portfolio value.

        Args:
            positions: Current portfolio positions
            market_data: Current market data for positions

        Returns:
            Decimal: Current portfolio value
        """
        portfolio_value = Decimal("0")

        # Create symbol-indexed market data for safe lookups
        market_by_symbol = {md.symbol: md for md in market_data}

        for position in positions:
            market = market_by_symbol.get(position.symbol)
            if market and position.symbol == market.symbol:
                # Calculate values without modifying position object
                current_price = market.close
                # unrealized_pnl = position.quantity * (current_price - position.entry_price)

                # Add position value to portfolio
                position_value = position.quantity * current_price
                portfolio_value += position_value

        return portfolio_value

    @time_execution
    async def _update_portfolio_history(self, portfolio_value: Decimal) -> None:
        """
        Update portfolio value history for risk calculations.

        Args:
            portfolio_value: Current portfolio value
        """
        self.portfolio_values.append(float(portfolio_value))

        # Calculate portfolio return if we have previous value
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2]
            if prev_value > 0:
                return_rate = (float(portfolio_value) - prev_value) / prev_value
                self.portfolio_returns.append(return_rate)

        # Keep only recent history
        # Use correlation_window or default to 252 days
        max_history = max(getattr(self.risk_config, "correlation_window", 30), 252)
        if len(self.portfolio_values) > max_history:
            self.portfolio_values = self.portfolio_values[-max_history:]

        if len(self.portfolio_returns) > max_history:
            self.portfolio_returns = self.portfolio_returns[-max_history:]

    @time_execution
    async def _calculate_var(self, days: int, portfolio_value: Decimal) -> Decimal:
        """
        Calculate Value at Risk for specified time horizon.

        Args:
            days: Time horizon in days
            portfolio_value: Current portfolio value

        Returns:
            Decimal: VaR value
        """
        if len(self.portfolio_returns) < 30:
            # Insufficient data, use conservative estimate scaled by sqrt(days)
            # This ensures 5-day VaR is larger than 1-day VaR
            base_var_pct = Decimal("0.02")  # 2% base VaR
            scaled_var_pct = base_var_pct * Decimal(str(np.sqrt(days)))
            return portfolio_value * scaled_var_pct

        # Calculate daily volatility
        returns_array = np.array(self.portfolio_returns)
        daily_volatility = np.std(returns_array)

        # Calculate VaR using normal distribution assumption
        # VaR = portfolio_value * volatility * sqrt(days) * z_score
        # Default VaR confidence level to 95%
        confidence_level = 0.95

        # Z-score for confidence level (90% = 1.282, 95% = 1.645, 99% = 2.326)
        if confidence_level == 0.90:
            z_score = 1.282
        elif confidence_level == 0.95:
            z_score = 1.645
        elif confidence_level == 0.99:
            z_score = 2.326
        else:
            # Use scipy.stats.norm.ppf for accurate z-score calculation
            from scipy.stats import norm

            z_score = norm.ppf(confidence_level)

        # Formula: VaR = portfolio_value * (volatility * sqrt(days) * z_score)
        var_percentage = daily_volatility * np.sqrt(days) * z_score
        var_value = portfolio_value * Decimal(str(var_percentage))

        return var_value

    @time_execution
    async def _calculate_expected_shortfall(self, portfolio_value: Decimal) -> Decimal:
        """
        Calculate Expected Shortfall (Conditional VaR).

        Args:
            portfolio_value: Current portfolio value

        Returns:
            Decimal: Expected shortfall value
        """
        if len(self.portfolio_returns) < 30:
            # Insufficient data, use conservative estimate
            return portfolio_value * Decimal("0.025")  # 2.5% ES

        # Calculate expected shortfall as average of worst returns
        returns_array = np.array(self.portfolio_returns)
        # Default VaR confidence level to 95%
        confidence_level = 0.95

        # Find threshold for worst (1-confidence_level) returns
        threshold = np.percentile(returns_array, (1 - confidence_level) * 100)

        # Calculate average of returns below threshold
        worst_returns = returns_array[returns_array <= threshold]

        if len(worst_returns) == 0:
            return portfolio_value * Decimal("0.02")  # Conservative fallback

        expected_shortfall = portfolio_value * Decimal(str(abs(np.mean(worst_returns))))

        return expected_shortfall

    @time_execution
    async def _calculate_max_drawdown(self) -> Decimal:
        """
        Calculate maximum historical drawdown.

        Returns:
            Decimal: Maximum drawdown value
        """
        if len(self.portfolio_values) < 2:
            return Decimal("0")

        # Calculate running maximum and drawdown
        running_max = np.maximum.accumulate(self.portfolio_values)
        drawdowns = (running_max - self.portfolio_values) / running_max

        max_drawdown = np.max(drawdowns)

        return Decimal(str(max_drawdown))

    @time_execution
    async def _calculate_current_drawdown(self, portfolio_value: Decimal) -> Decimal:
        """
        Calculate current drawdown from peak.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            Decimal: Current drawdown value
        """
        if len(self.portfolio_values) < 2:
            return Decimal("0")

        # Find peak value
        peak_value = max(self.portfolio_values)

        if peak_value <= 0:
            return Decimal("0")

        # Calculate current drawdown
        current_drawdown = (peak_value - float(portfolio_value)) / peak_value

        return Decimal(str(max(0, current_drawdown)))

    @time_execution
    async def _calculate_sharpe_ratio(self) -> Decimal | None:
        """
        Calculate Sharpe ratio for the portfolio.

        Returns:
            Optional[Decimal]: Sharpe ratio or None if insufficient data
        """
        if len(self.portfolio_returns) < 30:
            return None  # Insufficient data

        returns_array = np.array(self.portfolio_returns)

        # Calculate annualized metrics
        mean_return = np.mean(returns_array) * 252  # Annualize daily returns
        # Annualize daily volatility
        volatility = np.std(returns_array) * np.sqrt(252)

        if volatility == 0:
            return None

        # Risk-free rate (assume 0% for crypto)
        risk_free_rate = 0.0

        # Sharpe ratio = (return - risk_free_rate) / volatility
        sharpe_ratio = (mean_return - risk_free_rate) / volatility

        return Decimal(str(sharpe_ratio))

    @time_execution
    async def _determine_risk_level(
        self, var_1d: Decimal, current_drawdown: Decimal, sharpe_ratio: Decimal | None
    ) -> RiskLevel:
        """
        Determine risk level based on current metrics.

        Args:
            var_1d: 1-day Value at Risk
            current_drawdown: Current drawdown
            sharpe_ratio: Sharpe ratio

        Returns:
            RiskLevel: Determined risk level
        """
        # Get current portfolio value for percentage calculations
        current_portfolio_value = self.portfolio_values[-1] if self.portfolio_values else 1

        # Calculate VaR as percentage of portfolio value
        var_1d_pct = (
            var_1d / Decimal(str(current_portfolio_value))
            if current_portfolio_value > 0
            else Decimal("0")
        )

        # Risk level thresholds (as percentages)
        var_threshold_high = Decimal("0.05")  # 5% VaR
        var_threshold_critical = Decimal("0.10")  # 10% VaR
        drawdown_threshold_high = Decimal("0.10")  # 10% drawdown
        drawdown_threshold_critical = Decimal("0.20")  # 20% drawdown
        sharpe_threshold_low = Decimal("-1.0")  # Negative Sharpe ratio

        # Check for critical risk
        if var_1d_pct > var_threshold_critical or current_drawdown > drawdown_threshold_critical:
            return RiskLevel.CRITICAL

        # Check for high risk
        if (
            var_1d_pct > var_threshold_high
            or current_drawdown > drawdown_threshold_high
            or (sharpe_ratio and sharpe_ratio < sharpe_threshold_low)
        ):
            return RiskLevel.HIGH

        # Check for medium risk
        if (
            var_1d_pct > Decimal("0.02")  # 2% VaR
            or current_drawdown > Decimal("0.05")  # 5% drawdown
            or (sharpe_ratio and sharpe_ratio < Decimal("0.5"))
        ):
            return RiskLevel.MEDIUM

        # Default to low risk
        return RiskLevel.LOW

    @time_execution
    async def update_position_returns(self, symbol: str, price: float) -> None:
        """
        Update position return history for individual position risk.

        Args:
            symbol: Trading symbol
            price: Current price
        """
        if symbol not in self.position_returns:
            self.position_returns[symbol] = []
            self.position_prices[symbol] = []

        # Store the current price
        self.position_prices[symbol].append(price)

        # Calculate return if we have previous price
        if len(self.position_prices[symbol]) > 1:
            prev_price = self.position_prices[symbol][-2]
            if prev_price > 0:
                return_rate = (price - prev_price) / prev_price
                self.position_returns[symbol].append(return_rate)

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

        current_value = self.portfolio_values[-1] if self.portfolio_values else 0
        peak_value = max(self.portfolio_values) if self.portfolio_values else 0

        summary = {
            "current_portfolio_value": current_value,
            "peak_portfolio_value": peak_value,
            "total_return": (
                (current_value - self.portfolio_values[0]) / self.portfolio_values[0]
                if len(self.portfolio_values) > 1
                else 0
            ),
            "data_points": len(self.portfolio_values),
            "return_data_points": len(self.portfolio_returns),
            "position_symbols": list(self.position_returns.keys()),
        }

        return summary
