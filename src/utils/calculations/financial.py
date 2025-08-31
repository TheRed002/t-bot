"""Financial calculations for the T-Bot trading system."""

from decimal import Decimal
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.utils.decimal_utils import ZERO, safe_divide, to_decimal
from src.utils.interfaces import CalculatorInterface

logger = get_logger(__name__)

# Epsilon for Decimal comparisons to handle rounding precision issues
DECIMAL_EPSILON = Decimal("1e-10")


class FinancialCalculator(CalculatorInterface):
    """
    Class for all financial calculations.
    Uses caching to avoid redundant computations.
    Implements CalculatorInterface for dependency injection.

    Note: This class uses static methods with caching, so no instance
    is needed. The caching is shared across all uses.
    """

    @staticmethod
    @lru_cache(maxsize=128)
    def sharpe_ratio(
        returns: tuple[Decimal, ...],
        risk_free_rate: Decimal = Decimal("0.02"),
        periods_per_year: int = 252
    ) -> Decimal:
        """
        Calculate Sharpe ratio for given returns.

        Args:
            returns: Tuple of Decimal returns (cached for immutability)
            risk_free_rate: Annual risk-free rate as Decimal
            periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)

        Returns:
            Sharpe ratio as Decimal
        """
        if len(returns) < 2:
            return ZERO

        # Convert to Decimal for precise calculations
        returns_decimal = [to_decimal(r) for r in returns]
        rf_rate = to_decimal(risk_free_rate)
        periods_decimal = to_decimal(periods_per_year)

        # Convert annual risk-free rate to period rate
        period_rf_rate = rf_rate / periods_decimal

        # Calculate excess returns with Decimal precision
        excess_returns = [r - period_rf_rate for r in returns_decimal]
        n_returns = Decimal(len(excess_returns))

        # Calculate mean excess return
        mean_excess = sum(excess_returns) / n_returns

        # Calculate standard deviation with Decimal precision
        variance = sum((r - mean_excess) ** 2 for r in excess_returns) / (n_returns - Decimal("1"))
        if variance <= ZERO:
            return ZERO

        std_excess = variance.sqrt()

        if std_excess <= DECIMAL_EPSILON:
            return ZERO

        # Annualized Sharpe ratio
        return (mean_excess * periods_decimal) / (std_excess * periods_decimal.sqrt())

    @staticmethod
    @lru_cache(maxsize=128)
    def sortino_ratio(
        returns: tuple[Decimal, ...],
        risk_free_rate: Decimal = Decimal("0.02"),
        periods_per_year: int = 252
    ) -> Decimal:
        """Calculate Sortino ratio using math_utils implementation to avoid duplication."""
        from src.utils.math_utils import calculate_sortino_ratio

        # Convert tuple to list for math_utils compatibility
        return calculate_sortino_ratio(list(returns), risk_free_rate, periods_per_year)

    @staticmethod
    @lru_cache(maxsize=128)
    def calmar_ratio(returns: tuple[Decimal, ...], periods_per_year: int = 252) -> Decimal:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        Args:
            returns: Tuple of Decimal returns
            periods_per_year: Number of periods in a year

        Returns:
            Calmar ratio as Decimal
        """
        if len(returns) < 2:
            return ZERO

        # Convert to Decimal for precise calculations
        returns_decimal = [to_decimal(r) for r in returns]
        periods_decimal = to_decimal(periods_per_year)

        # Calculate cumulative returns with Decimal precision
        cumulative = [Decimal("1")]
        for r in returns_decimal:
            cumulative.append(cumulative[-1] * (Decimal("1") + r))
        cumulative = cumulative[1:]  # Remove initial 1

        # Calculate max drawdown
        running_max = [cumulative[0]]
        for i in range(1, len(cumulative)):
            running_max.append(max(running_max[-1], cumulative[i]))

        drawdown = [(cumulative[i] - running_max[i]) / running_max[i] for i in range(len(cumulative))]
        max_drawdown = abs(min(drawdown))

        if max_drawdown <= DECIMAL_EPSILON:
            return Decimal("inf")

        # Annualized return
        total_return = cumulative[-1] / cumulative[0] - Decimal("1")
        n_periods = Decimal(len(returns_decimal))

        # Calculate annualized return: (1 + total_return) ^ (periods_per_year / n_periods) - 1
        if total_return <= Decimal("-1"):
            return ZERO

        base = Decimal("1") + total_return
        exponent = periods_decimal / n_periods
        # Use Decimal power approximation to maintain precision
        # For power calculation with Decimal, we use the natural logarithm approach:
        # a^b = exp(b * ln(a))
        import math

        try:
            # Use high precision calculation maintaining Decimal precision
            # For a^b calculation with Decimals, use ln approximation carefully
            # Convert only when necessary and immediately back to Decimal
            ln_base_float = math.log(float(base))
            exp_result_float = math.exp(float(exponent) * ln_base_float)
            # Convert back to Decimal using string to preserve precision
            annualized_return = to_decimal(str(exp_result_float)) - Decimal("1")
        except (ValueError, OverflowError):
            # Fallback to simple approximation if exponential calculation fails
            annualized_return = ZERO

        return safe_divide(annualized_return, max_drawdown, ZERO)

    @staticmethod
    @lru_cache(maxsize=256)
    def moving_average(prices: tuple[Decimal, ...], period: int, ma_type: str = "simple") -> Decimal:
        """
        Calculate moving average (last value).

        Args:
            prices: Tuple of Decimal prices
            period: Period for moving average
            ma_type: Type of MA ('simple', 'exponential', 'weighted')

        Returns:
            Moving average value as Decimal
        """
        if len(prices) < period:
            if not prices:
                return ZERO
            return sum(prices) / Decimal(len(prices))

        prices_decimal = [to_decimal(p) for p in prices]
        period_decimal = Decimal(period)

        if ma_type == "simple":
            recent_prices = prices_decimal[-period:]
            return sum(recent_prices) / period_decimal

        elif ma_type == "exponential":
            recent_prices = prices_decimal[-period:]
            alpha = Decimal("2") / (period_decimal + Decimal("1"))

            # Calculate weights with Decimal precision
            weights = []
            for i in range(period - 1, -1, -1):
                weight = (Decimal("1") - alpha) ** i
                weights.append(weight)

            weights_sum = sum(weights)
            weighted_sum = sum(price * weight for price, weight in zip(recent_prices, weights, strict=False))
            return Decimal(str(weighted_sum / weights_sum))

        elif ma_type == "weighted":
            recent_prices = prices_decimal[-period:]
            weights_sum = Decimal("0")
            weighted_sum = Decimal("0")

            for i, price in enumerate(recent_prices):
                weight = Decimal(i + 1)
                weighted_sum += price * weight
                weights_sum += weight

            return weighted_sum / weights_sum

        else:
            raise ValidationError(f"Unknown MA type: {ma_type}", field_name="ma_type", field_value=ma_type)

    @staticmethod
    def max_drawdown(prices: list[Decimal] | NDArray[np.float64]) -> tuple[Decimal, int, int]:
        """
        Calculate maximum drawdown and duration.

        Args:
            prices: List of Decimal prices or numpy array

        Returns:
            Tuple of (max_drawdown_pct, start_idx, end_idx)
        """
        if isinstance(prices, np.ndarray):
            # Convert numpy array to list of Decimals using string conversion to preserve precision
            prices_decimal = [to_decimal(str(float(p))) for p in prices]
        else:
            prices_decimal = [to_decimal(p) for p in prices]

        if len(prices_decimal) < 2:
            return ZERO, 0, 0

        # Calculate cumulative max with Decimal precision
        cummax = [prices_decimal[0]]
        for price in prices_decimal[1:]:
            cummax.append(max(cummax[-1], price))

        # Calculate drawdown
        drawdown = []
        for i, price in enumerate(prices_decimal):
            if cummax[i] > ZERO:
                dd = (price - cummax[i]) / cummax[i]
            else:
                dd = ZERO
            drawdown.append(dd)

        # Find max drawdown
        max_dd = min(drawdown)
        max_dd_idx = drawdown.index(max_dd)

        # Find start of drawdown (last peak before max drawdown)
        start_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if abs(drawdown[i]) <= DECIMAL_EPSILON:
                start_idx = i
                break

        return abs(max_dd), int(start_idx), int(max_dd_idx)

    @staticmethod
    @lru_cache(maxsize=128)
    def kelly_criterion(
        win_probability: Decimal, win_amount: Decimal, loss_amount: Decimal, kelly_fraction: Decimal = Decimal("0.25")
    ) -> Decimal:
        """
        Calculate Kelly criterion for position sizing.

        Args:
            win_probability: Probability of winning as Decimal
            win_amount: Average win amount as Decimal
            loss_amount: Average loss amount (positive value) as Decimal
            kelly_fraction: Fraction of Kelly to use (default 0.25 for safety) as Decimal

        Returns:
            Fraction of capital to risk as Decimal
        """
        win_prob = to_decimal(win_probability)
        win_amt = to_decimal(win_amount)
        loss_amt = to_decimal(loss_amount)
        kelly_frac = to_decimal(kelly_fraction)

        if win_prob <= ZERO or win_prob >= Decimal("1"):
            return ZERO

        if win_amt <= ZERO or loss_amt <= ZERO:
            return ZERO

        # Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        q = Decimal("1") - win_prob
        b = safe_divide(win_amt, loss_amt, ZERO)

        if b <= ZERO:
            return ZERO

        kelly = safe_divide(win_prob * b - q, b, ZERO)

        # Apply Kelly fraction for safety
        kelly = kelly * kelly_frac

        # Limit to reasonable range
        max_kelly = Decimal("0.25")  # Max 25% of capital
        return max(ZERO, min(kelly, max_kelly))

    @staticmethod
    def position_size_volatility_adjusted(
        account_balance: Decimal,
        risk_per_trade: Decimal,
        stop_distance: Decimal,
        volatility: Decimal,
        target_volatility: Decimal = Decimal("0.02"),
    ) -> Decimal:
        """
        Calculate position size adjusted for volatility.

        Args:
            account_balance: Total account balance as Decimal
            risk_per_trade: Risk per trade (e.g., 0.02 for 2%) as Decimal
            stop_distance: Distance to stop loss as Decimal
            volatility: Current volatility as Decimal
            target_volatility: Target volatility for normalization as Decimal

        Returns:
            Position size as Decimal
        """
        balance = to_decimal(account_balance)
        risk = to_decimal(risk_per_trade)
        stop_dist = to_decimal(stop_distance)
        vol = to_decimal(volatility)
        target_vol = to_decimal(target_volatility)

        if stop_dist <= ZERO or vol <= ZERO:
            return ZERO

        # Basic position size with Decimal precision
        basic_size = safe_divide(balance * risk, stop_dist, ZERO)

        # Adjust for volatility
        volatility_adjustment = safe_divide(target_vol, vol, Decimal("1"))
        # Limit adjustment between 0.5 and 2.0
        min_adj = Decimal("0.5")
        max_adj = Decimal("2.0")
        volatility_adjustment = max(min_adj, min(volatility_adjustment, max_adj))

        adjusted_size = basic_size * volatility_adjustment

        # Round to 8 decimal places for crypto precision
        return adjusted_size.quantize(Decimal("0.00000001"))

    @staticmethod
    def calculate_returns(prices: list[Decimal] | NDArray[np.float64], method: str = "simple") -> list[Decimal]:
        """
        Calculate returns from prices with Decimal precision.

        Args:
            prices: List of Decimal prices or numpy array
            method: 'simple' or 'log' returns

        Returns:
            List of Decimal returns
        """
        if isinstance(prices, np.ndarray):
            # Convert numpy array to Decimal list using string conversion to preserve precision
            prices_decimal = [to_decimal(str(float(p))) for p in prices]
        else:
            prices_decimal = [to_decimal(p) for p in prices]

        if len(prices_decimal) < 2:
            return []

        returns: list[Decimal] = []
        if method == "simple":
            for i in range(1, len(prices_decimal)):
                if prices_decimal[i - 1] > ZERO:
                    ret = (prices_decimal[i] - prices_decimal[i - 1]) / prices_decimal[i - 1]
                    returns.append(ret)
                else:
                    returns.append(ZERO)
        elif method == "log":
            for i in range(1, len(prices_decimal)):
                if prices_decimal[i - 1] > ZERO and prices_decimal[i] > ZERO:
                    # Use natural log approximation for Decimal
                    ratio = prices_decimal[i] / prices_decimal[i - 1]
                    # Convert to float for log calculation, then back to Decimal via string
                    log_return = to_decimal(str(np.log(float(ratio))))
                    returns.append(log_return)
                else:
                    returns.append(ZERO)
        else:
            raise ValidationError(f"Unknown return method: {method}", field_name="method", field_value=method)

        return returns

    @staticmethod
    def risk_reward_ratio(entry_price: Decimal, stop_loss: Decimal, take_profit: Decimal) -> Decimal:
        """
        Calculate risk/reward ratio for a trade.

        Args:
            entry_price: Entry price as Decimal
            stop_loss: Stop loss price as Decimal
            take_profit: Take profit price as Decimal

        Returns:
            Risk/reward ratio as Decimal
        """
        entry = to_decimal(entry_price)
        stop = to_decimal(stop_loss)
        profit = to_decimal(take_profit)

        risk = abs(entry - stop)
        reward = abs(profit - entry)

        if risk <= DECIMAL_EPSILON:
            return Decimal("inf")

        return safe_divide(reward, risk, ZERO)

    @staticmethod
    @lru_cache(maxsize=128)
    def expected_value(win_probability: Decimal, avg_win: Decimal, avg_loss: Decimal) -> Decimal:
        """
        Calculate expected value of a trading strategy.

        Args:
            win_probability: Probability of winning as Decimal
            avg_win: Average win amount as Decimal
            avg_loss: Average loss amount (as positive number) as Decimal

        Returns:
            Expected value as Decimal
        """
        win_prob = to_decimal(win_probability)
        win_amount = to_decimal(avg_win)
        loss_amount = to_decimal(avg_loss)

        return (win_prob * win_amount) - ((Decimal("1") - win_prob) * loss_amount)

    @staticmethod
    def profit_factor(wins: list[Decimal], losses: list[Decimal]) -> Decimal:
        """
        Calculate profit factor (gross wins / gross losses).

        Args:
            wins: List of winning amounts as Decimal
            losses: List of losing amounts (as positive numbers) as Decimal

        Returns:
            Profit factor as Decimal
        """
        wins_decimal = [to_decimal(w) for w in wins] if wins else []
        losses_decimal = [to_decimal(loss) for loss in losses] if losses else []

        total_wins = sum(wins_decimal, ZERO) if wins_decimal else ZERO
        total_losses = sum(losses_decimal, ZERO) if losses_decimal else ZERO

        if total_losses <= DECIMAL_EPSILON:
            return Decimal("inf") if total_wins > ZERO else ZERO

        return safe_divide(total_wins, total_losses, ZERO)

    # Interface compliance methods for factory pattern
    @staticmethod
    def calculate_compound_return(principal: Decimal, rate: Decimal, periods: int) -> Decimal:
        """Calculate compound return for interface compliance."""
        from src.utils.decimal_utils import to_decimal

        principal_decimal = to_decimal(principal)
        rate_decimal = to_decimal(rate)

        # A = P(1 + r)^n - P
        compound_amount = principal_decimal * ((1 + rate_decimal) ** periods)
        return compound_amount - principal_decimal

    @staticmethod
    def calculate_sharpe_ratio(returns: list[Decimal], risk_free_rate: Decimal) -> Decimal:
        """Calculate Sharpe ratio for interface compliance."""
        # Delegate to math_utils to avoid duplication
        from src.utils.math_utils import calculate_sharpe_ratio as math_sharpe_ratio

        return math_sharpe_ratio(returns, risk_free_rate)


# FinancialCalculator registration is handled by service_registry.py


# Backward compatibility - get instance from DI container
def get_financial_calculator() -> FinancialCalculator:
    """Get FinancialCalculator instance from DI container with lazy initialization."""
    from src.core.dependency_injection import injector

    try:
        return injector.resolve("FinancialCalculator")
    except Exception as e:
        # Create fallback instance if DI container not available
        logger.debug(f"Failed to resolve FinancialCalculator from DI container: {e}")
        return FinancialCalculator()


calc = get_financial_calculator()
