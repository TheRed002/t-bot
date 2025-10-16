"""
Backtesting Metrics Calculator.

This module provides comprehensive metrics calculation for backtest results,
including risk-adjusted returns, drawdown analysis, and trade statistics.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.core.base.component import BaseComponent
from src.utils.decimal_utils import to_decimal
from src.utils.decorators import time_execution
from src.utils.financial_calculations import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var_cvar,
    calculate_volatility,
)
from src.utils.financial_constants import (
    DEFAULT_RISK_FREE_RATE,
)


class BacktestMetrics(BaseComponent):
    """Container for all backtest metrics."""

    def __init__(self) -> None:
        """Initialize metrics container."""
        super().__init__()  # Initialize BaseComponent
        self.metrics: dict[str, Any] = {}

    def add(self, name: str, value: Any) -> None:
        """Add a metric."""
        self.metrics[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        """Get a metric value."""
        return self.metrics.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.metrics.copy()


class MetricsCalculator:
    """
    Calculator for comprehensive backtest metrics.

    Provides calculation of:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Drawdown metrics
    - Trade statistics
    - Risk metrics (VaR, CVaR)
    """

    # Use constants from shared financial constants module

    def __init__(self, risk_free_rate: float | None = None) -> None:
        """
        Initialize metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = to_decimal(
            risk_free_rate if risk_free_rate is not None else DEFAULT_RISK_FREE_RATE
        )
        from src.core.logging import get_logger

        self.logger = get_logger(__name__)
        self.logger.info("MetricsCalculator initialized", risk_free_rate=risk_free_rate)

    @time_execution
    def calculate_all(
        self,
        equity_curve: list[dict[str, Any]],
        trades: list[dict[str, Any]],
        daily_returns: list[float],
        initial_capital: float,
    ) -> dict[str, Any]:
        """
        Calculate all metrics from backtest data.

        Args:
            equity_curve: List of equity values over time
            trades: List of executed trades
            daily_returns: Daily return series
            initial_capital: Initial capital amount

        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}

        # Return metrics
        if equity_curve:
            metrics.update(self._calculate_return_metrics(equity_curve, initial_capital))

        # Risk-adjusted metrics
        if daily_returns:
            metrics.update(self._calculate_risk_adjusted_metrics(daily_returns))

        # Drawdown metrics
        if equity_curve:
            metrics.update(self._calculate_drawdown_metrics(equity_curve))

        # Trade statistics
        if trades:
            metrics.update(self._calculate_trade_statistics(trades))

        # Risk metrics
        if daily_returns:
            metrics.update(self._calculate_risk_metrics(daily_returns, initial_capital))

        self.logger.info("Metrics calculated", num_metrics=len(metrics))
        return metrics

    def _calculate_return_metrics(
        self, equity_curve: list[dict[str, Any]], initial_capital: float
    ) -> dict[str, Any]:
        """Calculate return-based metrics."""
        if not equity_curve:
            return {}

        final_equity = to_decimal(equity_curve[-1]["equity"])
        initial_capital_dec = to_decimal(initial_capital)
        total_return = (final_equity - initial_capital_dec) / initial_capital_dec

        # Calculate annualized return
        start_date = equity_curve[0]["timestamp"]
        end_date = equity_curve[-1]["timestamp"]
        days = (end_date - start_date).days
        years = to_decimal(days) / to_decimal("365.25")

        if years > 0:
            annual_return = (to_decimal("1") + total_return) ** (
                to_decimal("1") / years
            ) - to_decimal("1")
        else:
            annual_return = to_decimal("0")

        return {
            "total_return": total_return * to_decimal("100"),
            "annual_return": annual_return * to_decimal("100"),
            "final_equity": final_equity,
        }

    def _calculate_risk_adjusted_metrics(self, daily_returns: list[float]) -> dict[str, Any]:
        """Calculate risk-adjusted performance metrics."""
        if not daily_returns:
            return {}

        returns_array = np.array(daily_returns)

        # Use shared calculations
        volatility = calculate_volatility(returns_array)
        sharpe_ratio = calculate_sharpe_ratio(returns_array, self.risk_free_rate)
        sortino_ratio = calculate_sortino_ratio(returns_array, self.risk_free_rate)

        return {
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "mean_daily_return": to_decimal(str(np.mean(returns_array[~np.isnan(returns_array)])))
            * to_decimal("100"),
            "std_daily_return": to_decimal(str(np.std(returns_array[~np.isnan(returns_array)])))
            * to_decimal("100"),
        }

    def _calculate_drawdown_metrics(self, equity_curve: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate drawdown-related metrics."""
        if not equity_curve:
            return {}

        # Use shared calculation for maximum drawdown
        max_drawdown, drawdown_info = calculate_max_drawdown(equity_curve)

        # Calculate additional metrics using original logic for compatibility
        df = pd.DataFrame(equity_curve)
        df.set_index("timestamp", inplace=True)

        # Calculate running maximum
        running_max = df["equity"].expanding().max()

        # Calculate drawdown
        drawdown = (df["equity"] - running_max) / running_max

        # Maximum drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0

        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0

        # Recovery factor (total return / max drawdown)
        if len(equity_curve) > 1:
            final_equity = to_decimal(equity_curve[-1]["equity"])
            initial_equity = to_decimal(equity_curve[0]["equity"])
            total_return = (final_equity - initial_equity) / initial_equity
            recovery_factor = total_return / max_drawdown if max_drawdown > 0 else to_decimal("0")
        else:
            recovery_factor = to_decimal("0")

        return {
            "max_drawdown": max_drawdown * to_decimal("100"),
            "max_drawdown_duration_days": max_duration,
            "recovery_factor": recovery_factor,
            "current_drawdown": to_decimal(str(abs(drawdown.iloc[-1]))) * to_decimal("100")
            if not drawdown.empty
            else to_decimal("0"),
        }

    def _calculate_trade_statistics(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate trade-related statistics."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }

        # Separate winning and losing trades
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]

        # Win rate
        win_rate = (
            to_decimal(len(winning_trades)) / to_decimal(len(trades)) if trades else to_decimal("0")
        )

        # Average win/loss
        avg_win = (
            to_decimal(str(np.mean([float(t["pnl"]) for t in winning_trades])))
            if winning_trades
            else to_decimal("0")
        )
        avg_loss = (
            to_decimal(str(abs(np.mean([float(t["pnl"]) for t in losing_trades]))))
            if losing_trades
            else to_decimal("0")
        )

        # Profit factor
        total_wins = sum(to_decimal(t["pnl"]) for t in winning_trades)
        total_losses = abs(sum(to_decimal(t["pnl"]) for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else to_decimal("999.99")

        # Payoff ratio
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else to_decimal("999.99")

        # Trade duration statistics
        durations = []
        for trade in trades:
            duration = to_decimal(
                str((trade["exit_time"] - trade["entry_time"]).total_seconds())
            ) / to_decimal("3600")  # Hours
            durations.append(duration)

        avg_duration = (
            to_decimal(str(np.mean([float(d) for d in durations])))
            if durations
            else to_decimal("0")
        )

        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade["pnl"] > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        return {
            "win_rate": win_rate * to_decimal("100"),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "payoff_ratio": payoff_ratio,
            "avg_trade_duration_hours": avg_duration,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "largest_win": (
                to_decimal(str(max([float(t["pnl"]) for t in trades])))
                if trades
                else to_decimal("0")
            ),
            "largest_loss": (
                to_decimal(str(abs(min([float(t["pnl"]) for t in trades]))))
                if trades
                else to_decimal("0")
            ),
        }

    def _calculate_risk_metrics(
        self, daily_returns: list[float], initial_capital: float
    ) -> dict[str, Any]:
        """Calculate risk metrics (VaR, CVaR, etc.)."""
        if not daily_returns:
            return {}

        returns_array = np.array(daily_returns)

        # Use shared calculations for VaR and CVaR
        var_95, cvar_95 = calculate_var_cvar(returns_array, confidence_level=0.95)

        # Convert to absolute values in terms of capital
        initial_capital_dec = to_decimal(initial_capital)
        var_95_capital = var_95 * initial_capital_dec
        cvar_95_capital = cvar_95 * initial_capital_dec

        # Maximum daily loss
        max_daily_loss = to_decimal(str(np.min(returns_array))) * initial_capital_dec

        # Skewness and Kurtosis (check for sufficient variance)
        if len(returns_array) < 3 or np.std(returns_array) == 0:
            # Need at least 3 data points for skewness, and non-zero variance
            skewness = to_decimal("0")
            kurtosis = to_decimal("0")
        else:
            skew_val = stats.skew(returns_array)
            kurt_val = stats.kurtosis(returns_array)
            # Handle NaN from scipy.stats
            skewness = to_decimal("0") if np.isnan(skew_val) else to_decimal(str(skew_val))
            kurtosis = to_decimal("0") if np.isnan(kurt_val) else to_decimal(str(kurt_val))

        # Omega ratio (probability-weighted ratio of gains vs losses)
        threshold_return = 0
        gains = returns_array[returns_array > threshold_return] - threshold_return
        losses = threshold_return - returns_array[returns_array <= threshold_return]

        if len(losses) > 0 and np.sum(losses) > 0:
            omega_ratio = to_decimal(str(np.sum(gains))) / to_decimal(str(np.sum(losses)))
        else:
            omega_ratio = to_decimal("999.99") if len(gains) > 0 else to_decimal("0")

        return {
            "var_95": abs(var_95_capital),
            "cvar_95": abs(cvar_95_capital),
            "max_daily_loss": abs(max_daily_loss),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "omega_ratio": omega_ratio,
        }

    def calculate_rolling_metrics(
        self,
        equity_curve: list[dict[str, Any]],
        window: int = 30,
    ) -> pd.DataFrame:
        """
        Calculate rolling metrics over time.

        Args:
            equity_curve: Equity curve data
            window: Rolling window size in days

        Returns:
            DataFrame with rolling metrics
        """
        if not equity_curve:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(equity_curve)
        df.set_index("timestamp", inplace=True)

        # Calculate daily returns
        df["returns"] = df["equity"].pct_change()

        # Rolling metrics
        df["rolling_return"] = df["returns"].rolling(window).mean() * 252
        df["rolling_volatility"] = df["returns"].rolling(window).std() * np.sqrt(252)
        df["rolling_sharpe"] = (df["rolling_return"] - float(self.risk_free_rate)) / df[
            "rolling_volatility"
        ]

        # Rolling drawdown
        rolling_max = df["equity"].rolling(window, min_periods=1).max()
        df["rolling_drawdown"] = (df["equity"] - rolling_max) / rolling_max

        return df[["rolling_return", "rolling_volatility", "rolling_sharpe", "rolling_drawdown"]]
