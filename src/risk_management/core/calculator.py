"""Centralized risk calculations with caching to eliminate duplication."""

from datetime import datetime
from decimal import Decimal

import numpy as np

from src.base import BaseComponent
from src.core.types.market import MarketData
from src.core.types.risk import RiskLevel, RiskMetrics
from src.core.types.trading import Position


class RiskCalculator(BaseComponent):
    """
    Centralized risk calculator with caching.

    This eliminates duplication of risk calculations across modules
    by providing a single source of truth for all risk metrics.
    """

    def __init__(self):
        """Initialize risk calculator."""
        super().__init__()  # Initialize BaseComponent
        self._cache = {}  # Cache TTL in seconds

        # Historical data storage
        self.portfolio_values: list[float] = []
        self.portfolio_returns: list[float] = []
        self.position_data: dict[str, list[float]] = {}

    def calculate_var(
        self, returns: list[float], confidence_level: float = 0.95, time_horizon: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days

        Returns:
            VaR value
        """
        if not returns or len(returns) < 2:
            return 0.0

        # Convert to numpy array for calculations
        returns_array = np.array(returns)

        # Calculate VaR using percentile method
        var_percentile = (1 - confidence_level) * 100
        var_daily = np.percentile(returns_array, var_percentile)

        # Scale to time horizon
        var_scaled = var_daily * np.sqrt(time_horizon)

        return abs(var_scaled)

    def calculate_expected_shortfall(
        self, returns: list[float], confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).

        Args:
            returns: Historical returns
            confidence_level: Confidence level

        Returns:
            Expected shortfall value
        """
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns_array, var_percentile)

        # Calculate mean of returns below VaR threshold
        tail_returns = returns_array[returns_array <= var_threshold]

        if len(tail_returns) == 0:
            return abs(var_threshold)

        return abs(np.mean(tail_returns))

    def calculate_sharpe_ratio(self, returns: list[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Historical returns
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate

        if np.std(returns_array) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(returns_array)

    def calculate_sortino_ratio(
        self, returns: list[float], risk_free_rate: float = 0.02, target_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino ratio.

        Args:
            returns: Historical returns
            risk_free_rate: Risk-free rate
            target_return: Target return for downside deviation

        Returns:
            Sortino ratio
        """
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate

        # Calculate downside deviation
        downside_returns = np.minimum(returns_array - target_return, 0)
        downside_deviation = np.std(downside_returns)

        if downside_deviation == 0:
            return 0.0

        return np.mean(excess_returns) / downside_deviation

    def calculate_max_drawdown(self, values: list[float]) -> tuple[float, int, int]:
        """
        Calculate maximum drawdown.

        Args:
            values: Portfolio values

        Returns:
            Tuple of (max_drawdown, peak_index, trough_index)
        """
        if not values or len(values) < 2:
            return 0.0, 0, 0

        values_array = np.array(values)
        cumulative_returns = (1 + values_array).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max

        max_drawdown = np.min(drawdown)
        trough_index = np.argmin(drawdown)

        # Find the peak before the trough
        peak_index = np.argmax(cumulative_returns[: trough_index + 1])

        return abs(max_drawdown), peak_index, trough_index

    def calculate_calmar_ratio(self, returns: list[float], period_years: float = 1.0) -> float:
        """
        Calculate Calmar ratio.

        Args:
            returns: Historical returns
            period_years: Period in years

        Returns:
            Calmar ratio
        """
        if not returns:
            return 0.0

        annual_return = np.mean(returns) * 252  # Assuming daily returns
        max_dd, _, _ = self.calculate_max_drawdown(returns)

        if max_dd == 0:
            return 0.0

        return annual_return / abs(max_dd)

    def calculate_portfolio_metrics(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.

        Args:
            positions: Current positions
            market_data: Market data for positions

        Returns:
            Complete risk metrics
        """
        # Calculate portfolio value and total exposure with safe symbol matching
        market_by_symbol = {md.symbol: md for md in market_data}

        portfolio_value = sum(
            float(pos.quantity) * float(market_by_symbol[pos.symbol].close)
            for pos in positions
            if pos.symbol in market_by_symbol
        )

        # Total exposure is the sum of absolute position values
        total_exposure = sum(
            abs(float(pos.quantity)) * float(market_by_symbol[pos.symbol].close)
            for pos in positions
            if pos.symbol in market_by_symbol
        )

        # Update historical data
        self.portfolio_values.append(portfolio_value)

        # Calculate returns if we have enough history
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
            self.portfolio_returns = returns.tolist()

        # Calculate all metrics
        var_1d = self.calculate_var(self.portfolio_returns, 0.95, 1)
        var_5d = self.calculate_var(self.portfolio_returns, 0.95, 5)
        expected_shortfall = self.calculate_expected_shortfall(self.portfolio_returns)
        sharpe = self.calculate_sharpe_ratio(self.portfolio_returns)
        sortino = self.calculate_sortino_ratio(self.portfolio_returns)
        max_dd, _, _ = self.calculate_max_drawdown(self.portfolio_returns)
        calmar = self.calculate_calmar_ratio(self.portfolio_returns)

        # Determine risk level
        risk_level = self._determine_risk_level(var_1d, max_dd, sharpe)

        return RiskMetrics(
            timestamp=datetime.utcnow(),
            portfolio_value=Decimal(str(portfolio_value)),
            total_exposure=Decimal(str(total_exposure)),
            var_1d=Decimal(str(var_1d)),
            var_5d=Decimal(str(var_5d)),
            expected_shortfall=Decimal(str(expected_shortfall)),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=Decimal(str(max_dd)),
            current_drawdown=Decimal(str(self._calculate_current_drawdown())),
            calmar_ratio=calmar,
            risk_level=risk_level,
            position_count=len(positions),
            correlation_risk=self._calculate_correlation_risk(positions),
        )

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return 0.0

        peak = max(self.portfolio_values)
        current = self.portfolio_values[-1]

        if peak == 0:
            return 0.0

        return (peak - current) / peak

    def _calculate_correlation_risk(self, positions: list[Position]) -> Decimal:
        """Calculate correlation risk between positions."""
        if len(positions) < 2:
            return Decimal("0")

        # Simplified correlation risk calculation
        # In production, this would use actual correlation matrices
        unique_symbols = len(set(pos.symbol for pos in positions))
        concentration = 1.0 / unique_symbols if unique_symbols > 0 else 1.0

        return Decimal(str(1 - concentration))

    def _determine_risk_level(self, var: float, max_dd: float, sharpe: float) -> RiskLevel:
        """
        Determine overall risk level based on metrics.

        Args:
            var: Value at Risk
            max_dd: Maximum drawdown
            sharpe: Sharpe ratio

        Returns:
            Risk level classification
        """
        risk_score = 0

        # VaR scoring
        if var > 0.1:  # > 10%
            risk_score += 3
        elif var > 0.05:  # > 5%
            risk_score += 2
        elif var > 0.02:  # > 2%
            risk_score += 1

        # Drawdown scoring
        if max_dd > 0.2:  # > 20%
            risk_score += 3
        elif max_dd > 0.1:  # > 10%
            risk_score += 2
        elif max_dd > 0.05:  # > 5%
            risk_score += 1

        # Sharpe ratio scoring (inverse)
        if sharpe < 0.5:
            risk_score += 3
        elif sharpe < 1.0:
            risk_score += 2
        elif sharpe < 1.5:
            risk_score += 1

        # Map score to risk level
        if risk_score >= 7:
            return RiskLevel.CRITICAL
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def clear_cache(self) -> None:
        """Clear calculation cache."""
        self._cache.clear()
        self.logger.debug("Risk calculation cache cleared")

    def update_history(self, symbol: str, price: float, return_value: float | None = None) -> None:
        """
        Update historical data for a symbol.

        Args:
            symbol: Trading symbol
            price: Current price
            return_value: Return value if calculated
        """
        if symbol not in self.position_data:
            self.position_data[symbol] = []

        self.position_data[symbol].append(price)

        # Keep only recent history (e.g., last 1000 points)
        if len(self.position_data[symbol]) > 1000:
            self.position_data[symbol] = self.position_data[symbol][-1000:]


# Global instance for singleton pattern
_risk_calculator = RiskCalculator()


def get_risk_calculator() -> RiskCalculator:
    """Get the global risk calculator instance."""
    return _risk_calculator
