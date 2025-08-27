"""Financial calculations for the T-Bot trading system."""

from decimal import Decimal
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

# Epsilon for float comparisons to handle floating-point precision issues
EPSILON = 1e-10


class FinancialCalculator:
    """
    Class for all financial calculations.
    Uses caching to avoid redundant computations.

    Note: This class uses static methods with caching, so no instance
    is needed. The caching is shared across all uses.
    """

    @staticmethod
    @lru_cache(maxsize=128)
    def sharpe_ratio(
        returns: tuple[float, ...], risk_free_rate: float = 0.02, periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio for given returns.

        Args:
            returns: Tuple of returns (cached for immutability)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)

        Returns:
            Sharpe ratio
        """
        returns_array = np.array(returns)
        if len(returns_array) < 2:
            return 0.0

        # Convert annual risk-free rate to period rate
        period_rf_rate = risk_free_rate / periods_per_year

        # Calculate excess returns
        excess_returns = returns_array - period_rf_rate

        # Annualized Sharpe ratio
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)

        if abs(std_excess) < EPSILON:
            return 0.0

        return float((mean_excess / std_excess) * np.sqrt(periods_per_year))

    @staticmethod
    @lru_cache(maxsize=128)
    def sortino_ratio(
        returns: tuple[float, ...], risk_free_rate: float = 0.02, periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).

        Args:
            returns: Tuple of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year

        Returns:
            Sortino ratio
        """
        returns_array = np.array(returns)
        if len(returns_array) < 2:
            return 0.0

        period_rf_rate = risk_free_rate / periods_per_year
        excess_returns = returns_array - period_rf_rate

        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float("inf")  # No downside risk

        downside_std = np.std(downside_returns)

        if abs(downside_std) < EPSILON:
            return 0.0

        mean_excess = np.mean(excess_returns)
        return float((mean_excess / downside_std) * np.sqrt(periods_per_year))

    @staticmethod
    @lru_cache(maxsize=128)
    def calmar_ratio(returns: tuple[float, ...], periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        Args:
            returns: Tuple of returns
            periods_per_year: Number of periods in a year

        Returns:
            Calmar ratio
        """
        returns_array = np.array(returns)
        if len(returns_array) < 2:
            return 0.0

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns_array)

        # Calculate max drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))

        if abs(max_drawdown) < EPSILON:
            return float("inf")

        # Annualized return
        total_return = cumulative[-1] / cumulative[0] - 1
        n_periods = len(returns_array)
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

        return float(annualized_return / max_drawdown)

    @staticmethod
    @lru_cache(maxsize=256)
    def moving_average(prices: tuple[float, ...], period: int, ma_type: str = "simple") -> float:
        """
        Calculate moving average (last value).

        Args:
            prices: Tuple of prices
            period: Period for moving average
            ma_type: Type of MA ('simple', 'exponential', 'weighted')

        Returns:
            Moving average value
        """
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0.0

        if ma_type == "simple":
            return sum(prices[-period:]) / period

        elif ma_type == "exponential":
            prices_array = np.array(prices[-period:])
            alpha = 2 / (period + 1)
            weights = np.array([(1 - alpha) ** i for i in range(period - 1, -1, -1)])
            weights = weights / weights.sum()
            return float(np.sum(prices_array * weights))

        elif ma_type == "weighted":
            prices_array = np.array(prices[-period:])
            weights = np.arange(1, period + 1)
            return float(np.sum(prices_array * weights) / weights.sum())

        else:
            raise ValueError(f"Unknown MA type: {ma_type}")

    @staticmethod
    def max_drawdown(prices: list[float] | NDArray[np.float64]) -> tuple[float, int, int]:
        """
        Calculate maximum drawdown and duration.

        Args:
            prices: List or array of prices

        Returns:
            Tuple of (max_drawdown_pct, start_idx, end_idx)
        """
        prices_array = np.array(prices)
        if len(prices_array) < 2:
            return 0.0, 0, 0

        # Calculate cumulative max
        cummax = np.maximum.accumulate(prices_array)

        # Calculate drawdown
        drawdown = (prices_array - cummax) / cummax

        # Find max drawdown
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)

        # Find start of drawdown (last peak before max drawdown)
        start_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if abs(drawdown[i]) < EPSILON:
                start_idx = i
                break

        return float(abs(max_dd)), int(start_idx), int(max_dd_idx)

    @staticmethod
    @lru_cache(maxsize=128)
    def kelly_criterion(
        win_probability: float, win_amount: float, loss_amount: float, kelly_fraction: float = 0.25
    ) -> float:
        """
        Calculate Kelly criterion for position sizing.

        Args:
            win_probability: Probability of winning
            win_amount: Average win amount
            loss_amount: Average loss amount (positive value)
            kelly_fraction: Fraction of Kelly to use (default 0.25 for safety)

        Returns:
            Fraction of capital to risk
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0.0

        if win_amount <= 0 or loss_amount <= 0:
            return 0.0

        # Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        q = 1 - win_probability
        b = win_amount / loss_amount

        kelly = (win_probability * b - q) / b

        # Apply Kelly fraction for safety
        kelly = kelly * kelly_fraction

        # Limit to reasonable range
        return max(0.0, min(kelly, 0.25))  # Max 25% of capital

    @staticmethod
    def position_size_volatility_adjusted(
        account_balance: Decimal,
        risk_per_trade: float,
        stop_distance: float,
        volatility: float,
        target_volatility: float = 0.02,
    ) -> Decimal:
        """
        Calculate position size adjusted for volatility.

        Args:
            account_balance: Total account balance
            risk_per_trade: Risk per trade (e.g., 0.02 for 2%)
            stop_distance: Distance to stop loss as decimal
            volatility: Current volatility
            target_volatility: Target volatility for normalization

        Returns:
            Position size
        """
        if stop_distance <= 0 or volatility <= 0:
            return Decimal("0")

        # Basic position size
        basic_size = (float(account_balance) * risk_per_trade) / stop_distance

        # Adjust for volatility
        volatility_adjustment = target_volatility / volatility
        volatility_adjustment = max(0.5, min(volatility_adjustment, 2.0))  # Limit adjustment

        adjusted_size = basic_size * volatility_adjustment

        return Decimal(str(round(adjusted_size, 8)))

    @staticmethod
    def calculate_returns(
        prices: list[float] | NDArray[np.float64], method: str = "simple"
    ) -> NDArray[np.float64]:
        """
        Calculate returns from prices.

        Args:
            prices: List or array of prices
            method: 'simple' or 'log' returns

        Returns:
            Array of returns
        """
        prices_array = np.array(prices, dtype=np.float64)

        if len(prices_array) < 2:
            return np.array([], dtype=np.float64)

        returns: NDArray[np.float64]
        if method == "simple":
            returns = np.diff(prices_array) / prices_array[:-1]
        elif method == "log":
            returns = np.diff(np.log(prices_array))
        else:
            raise ValueError(f"Unknown return method: {method}")

        return returns.astype(np.float64)

    @staticmethod
    def risk_reward_ratio(entry_price: float, stop_loss: float, take_profit: float) -> float:
        """
        Calculate risk/reward ratio for a trade.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Risk/reward ratio
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        if abs(risk) < EPSILON:
            return float("inf")

        return reward / risk

    @staticmethod
    @lru_cache(maxsize=128)
    def expected_value(win_probability: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate expected value of a trading strategy.

        Args:
            win_probability: Probability of winning
            avg_win: Average win amount
            avg_loss: Average loss amount (as positive number)

        Returns:
            Expected value
        """
        return (win_probability * avg_win) - ((1 - win_probability) * avg_loss)

    @staticmethod
    def profit_factor(wins: list[float], losses: list[float]) -> float:
        """
        Calculate profit factor (gross wins / gross losses).

        Args:
            wins: List of winning amounts
            losses: List of losing amounts (as positive numbers)

        Returns:
            Profit factor
        """
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 0

        if abs(total_losses) < EPSILON:
            return float("inf") if total_wins > 0 else 0.0

        return total_wins / total_losses


# Create singleton instance
calc = FinancialCalculator()
