"""
Strategy Metrics - Comprehensive performance tracking and analysis.

This module provides:
- Real-time performance metrics
- Risk-adjusted return calculations
- Drawdown analysis
- Signal quality metrics
- Portfolio performance tracking
- Benchmark comparisons
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.core.types import Signal
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class PerformanceMetrics(BaseModel):
    """Comprehensive performance metrics for strategies."""

    # Basic performance
    total_return: Decimal = Field(default=Decimal("0"), description="Total return percentage")
    annual_return: Decimal = Field(default=Decimal("0"), description="Annualized return percentage")
    excess_return: Decimal = Field(default=Decimal("0"), description="Return above benchmark")

    # Risk metrics
    volatility: float = Field(default=0.0, description="Strategy volatility (annualized)")
    sharpe_ratio: float = Field(default=0.0, description="Sharpe ratio")
    sortino_ratio: float = Field(default=0.0, description="Sortino ratio")
    calmar_ratio: float = Field(default=0.0, description="Calmar ratio")

    # Drawdown metrics
    max_drawdown: Decimal = Field(default=Decimal("0"), description="Maximum drawdown")
    current_drawdown: Decimal = Field(default=Decimal("0"), description="Current drawdown")
    drawdown_duration: int = Field(default=0, description="Current drawdown duration in days")
    avg_drawdown_duration: float = Field(default=0.0, description="Average drawdown duration")

    # Trade statistics
    total_trades: int = Field(default=0, description="Total number of trades")
    winning_trades: int = Field(default=0, description="Number of winning trades")
    losing_trades: int = Field(default=0, description="Number of losing trades")
    win_rate: float = Field(default=0.0, description="Win rate percentage")
    profit_factor: float = Field(default=0.0, description="Profit factor")

    # Signal metrics
    total_signals: int = Field(default=0, description="Total signals generated")
    valid_signals: int = Field(default=0, description="Valid signals after validation")
    executed_signals: int = Field(default=0, description="Signals that were executed")
    signal_quality_score: float = Field(default=0.0, description="Overall signal quality (0-1)")

    # Timing metrics
    avg_trade_duration: float = Field(default=0.0, description="Average trade duration in hours")
    avg_signal_latency: float = Field(
        default=0.0, description="Average signal processing latency in ms"
    )

    # Risk management
    var_95: Decimal = Field(default=Decimal("0"), description="Value at Risk (95%)")
    cvar_95: Decimal = Field(default=Decimal("0"), description="Conditional Value at Risk (95%)")
    beta: float = Field(default=0.0, description="Beta relative to benchmark")

    # Metadata
    calculation_time: datetime = Field(default_factory=datetime.now)
    data_quality_score: float = Field(default=1.0, description="Data quality score (0-1)")


class MetricsCalculator:
    """
    Calculator for strategy performance metrics.

    Provides comprehensive analysis of strategy performance including:
    - Risk-adjusted returns
    - Drawdown analysis
    - Signal quality metrics
    - Benchmark comparisons
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize metrics calculator.

        Args:
            config: Calculator configuration
        """
        self.config = config or {}
        self._logger = logger

        # Risk-free rate for Sharpe calculation (default 2% annually)
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)

        # Trading days per year
        self.trading_days_per_year = self.config.get("trading_days_per_year", 252)

        # Benchmark data (if available)
        self._benchmark_returns: list[float] = []

    @time_execution
    async def calculate_comprehensive_metrics(
        self,
        equity_curve: list[dict[str, Any]],
        trades: list[dict[str, Any]],
        signals: list[Signal],
        initial_capital: float,
        benchmark_returns: list[float] | None = None,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            equity_curve: List of equity values over time
            trades: List of completed trades
            signals: List of generated signals
            initial_capital: Initial capital amount
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        try:
            if benchmark_returns:
                self._benchmark_returns = benchmark_returns

            # Calculate basic metrics
            basic_metrics = await self._calculate_basic_metrics(equity_curve, initial_capital)

            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(equity_curve, trades)

            # Calculate drawdown metrics
            drawdown_metrics = await self._calculate_drawdown_metrics(equity_curve)

            # Calculate trade statistics
            trade_metrics = await self._calculate_trade_metrics(trades)

            # Calculate signal metrics
            signal_metrics = await self._calculate_signal_metrics(signals, trades)

            # Calculate timing metrics
            timing_metrics = await self._calculate_timing_metrics(trades, signals)

            # Combine all metrics
            metrics = PerformanceMetrics(
                **basic_metrics,
                **risk_metrics,
                **drawdown_metrics,
                **trade_metrics,
                **signal_metrics,
                **timing_metrics,
            )

            self._logger.debug("Comprehensive metrics calculated successfully")
            return metrics

        except Exception as e:
            self._logger.error(f"Error calculating comprehensive metrics: {e}")
            return PerformanceMetrics()  # Return empty metrics on error

    async def _calculate_basic_metrics(
        self, equity_curve: list[dict[str, Any]], initial_capital: float
    ) -> dict[str, Any]:
        """Calculate basic performance metrics."""
        if not equity_curve:
            return {}

        final_equity = equity_curve[-1]["equity"]
        total_return = Decimal((final_equity - initial_capital) / initial_capital * 100)

        # Calculate time period for annualization
        start_time = equity_curve[0]["timestamp"]
        end_time = equity_curve[-1]["timestamp"]
        time_period_years = (end_time - start_time).days / 365.25

        # Annualized return
        if time_period_years > 0:
            annual_return = Decimal(
                ((final_equity / initial_capital) ** (1 / time_period_years) - 1) * 100
            )
        else:
            annual_return = total_return

        return {
            "total_return": total_return,
            "annual_return": annual_return,
        }

    async def _calculate_risk_metrics(
        self, equity_curve: list[dict[str, Any]], trades: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate risk-adjusted metrics."""
        if len(equity_curve) < 2:
            return {}

        # Calculate returns
        returns_list: list[float] = []
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i - 1]["equity"]
            curr_equity = equity_curve[i]["equity"]
            if prev_equity > 0:
                returns_list.append((curr_equity - prev_equity) / prev_equity)

        if not returns_list:
            return {}

        returns = np.array(returns_list)

        # Volatility (annualized)
        volatility = float(np.std(returns) * np.sqrt(self.trading_days_per_year))

        # Use FinancialCalculator for Sharpe and Sortino ratios
        from decimal import Decimal

        from src.utils.calculations.financial import FinancialCalculator

        # Convert to Decimal tuple for FinancialCalculator
        returns_decimal = tuple(Decimal(str(r)) for r in returns)

        # Calculate Sharpe ratio using FinancialCalculator
        sharpe_decimal = FinancialCalculator.sharpe_ratio(
            returns_decimal,
            risk_free_rate=Decimal(str(self.risk_free_rate)),
            periods_per_year=self.trading_days_per_year
        )
        sharpe_ratio = float(sharpe_decimal)

        # Calculate Sortino ratio using FinancialCalculator
        sortino_decimal = FinancialCalculator.sortino_ratio(
            returns_decimal,
            risk_free_rate=Decimal(str(self.risk_free_rate)),
            periods_per_year=self.trading_days_per_year
        )
        sortino_ratio = float(sortino_decimal)

        # VaR and CVaR (95%)
        var_95 = Decimal(str(np.percentile(returns, 5) * 100))  # 5th percentile for 95% VaR
        cvar_95 = Decimal(str(np.mean(returns[returns <= np.percentile(returns, 5)]) * 100))

        # Beta (if benchmark available)
        beta = 0.0
        if self._benchmark_returns and len(self._benchmark_returns) == len(returns):
            benchmark_returns = np.array(self._benchmark_returns[: len(returns)])
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = float(covariance / benchmark_variance) if benchmark_variance > 0 else 0.0

        return {
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "beta": beta,
        }

    async def _calculate_drawdown_metrics(
        self, equity_curve: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate drawdown-related metrics."""
        if not equity_curve:
            return {}

        equity_values = [point["equity"] for point in equity_curve]
        [point["timestamp"] for point in equity_curve]

        # Calculate running maximum (peak)
        running_max = []
        current_max = equity_values[0]

        for equity in equity_values:
            current_max = max(current_max, equity)
            running_max.append(current_max)

        # Calculate drawdowns
        drawdowns = []
        for i, equity in enumerate(equity_values):
            if running_max[i] > 0:
                drawdown = (running_max[i] - equity) / running_max[i] * 100
                drawdowns.append(drawdown)
            else:
                drawdowns.append(0)

        # Maximum drawdown
        max_drawdown = Decimal(str(max(drawdowns))) if drawdowns else Decimal("0")

        # Current drawdown
        current_drawdown = Decimal(str(drawdowns[-1])) if drawdowns else Decimal("0")

        # Drawdown duration analysis
        drawdown_periods = []
        current_period_start = None

        for i, dd in enumerate(drawdowns):
            if dd > 0:  # In drawdown
                if current_period_start is None:
                    current_period_start = i
            else:  # Not in drawdown
                if current_period_start is not None:
                    period_length = i - current_period_start
                    drawdown_periods.append(period_length)
                    current_period_start = None

        # Current drawdown duration
        current_drawdown_duration = 0
        if current_period_start is not None:
            current_drawdown_duration = len(drawdowns) - current_period_start

        # Average drawdown duration
        avg_drawdown_duration = float(np.mean(drawdown_periods)) if drawdown_periods else 0.0

        # Calmar ratio (annual return / max drawdown)
        if max_drawdown > 0:
            # Need annual return for this calculation
            # This would be calculated in the basic metrics
            pass

        return {
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "drawdown_duration": current_drawdown_duration,
            "avg_drawdown_duration": avg_drawdown_duration,
        }

    async def _calculate_trade_metrics(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate trade-related statistics."""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }

        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
        losing_trades = total_trades - winning_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Profit factor using FinancialCalculator
        from decimal import Decimal

        from src.utils.calculations.financial import FinancialCalculator

        winning_pnls = [Decimal(str(trade.get("pnl", 0))) for trade in trades if trade.get("pnl", 0) > 0]
        losing_pnls = [Decimal(str(abs(trade.get("pnl", 0)))) for trade in trades if trade.get("pnl", 0) < 0]

        profit_factor_decimal = FinancialCalculator.profit_factor(winning_pnls, losing_pnls)
        profit_factor = float(profit_factor_decimal)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    async def _calculate_signal_metrics(
        self, signals: list[Signal], trades: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate signal quality metrics."""
        total_signals = len(signals)

        if total_signals == 0:
            return {
                "total_signals": 0,
                "valid_signals": 0,
                "executed_signals": 0,
                "signal_quality_score": 0.0,
            }

        # Count valid signals (those with reasonable strength)
        valid_signals = sum(1 for signal in signals if signal.strength >= Decimal("0.5"))

        # Executed signals (approximate - signals that led to trades)
        executed_signals = len(trades)  # Simplified assumption

        # Signal quality score
        strength_scores = [float(signal.strength) for signal in signals]
        avg_strength = np.mean(strength_scores) if strength_scores else 0.0

        # Factor in execution rate
        execution_rate = executed_signals / total_signals if total_signals > 0 else 0.0

        # Combined quality score
        signal_quality_score = avg_strength * 0.7 + execution_rate * 0.3

        return {
            "total_signals": total_signals,
            "valid_signals": valid_signals,
            "executed_signals": executed_signals,
            "signal_quality_score": float(signal_quality_score),
        }

    async def _calculate_timing_metrics(
        self, trades: list[dict[str, Any]], signals: list[Signal]
    ) -> dict[str, Any]:
        """Calculate timing-related metrics."""
        # Average trade duration
        trade_durations = []
        for trade in trades:
            if "entry_time" in trade and "exit_time" in trade:
                duration = (
                    trade["exit_time"] - trade["entry_time"]
                ).total_seconds() / 3600  # hours
                trade_durations.append(duration)

        avg_trade_duration = float(np.mean(trade_durations)) if trade_durations else 0.0

        # Signal latency (placeholder - would need actual processing timestamps)
        avg_signal_latency = 0.0  # Would be calculated from signal processing times

        return {
            "avg_trade_duration": avg_trade_duration,
            "avg_signal_latency": avg_signal_latency,
        }


class RealTimeMetricsTracker:
    """
    Real-time metrics tracker for live strategy monitoring.

    Provides continuous monitoring and updating of strategy performance metrics
    during live trading operations.
    """

    def __init__(self, strategy_id: str, config: dict[str, Any] | None = None):
        """
        Initialize real-time metrics tracker.

        Args:
            strategy_id: Unique strategy identifier
            config: Tracker configuration
        """
        self.strategy_id = strategy_id
        self.config = config or {}
        self._logger = get_logger(f"{__name__}.MetricsTracker_{strategy_id}")

        # Current metrics
        self._current_metrics = PerformanceMetrics()
        self._calculator = MetricsCalculator(config)

        # Data collection
        self._equity_points: list[dict[str, Any]] = []
        self._trade_history: list[dict[str, Any]] = []
        self._signal_history: list[Signal] = []

        # Update frequency
        self._update_interval = timedelta(seconds=self.config.get("update_interval_seconds", 60))
        self._last_update = datetime.now(timezone.utc)

        # Limits for data retention
        self._max_equity_points = self.config.get("max_equity_points", 10000)
        self._max_trade_history = self.config.get("max_trade_history", 1000)
        self._max_signal_history = self.config.get("max_signal_history", 1000)

    async def update_equity(self, equity: float, timestamp: datetime | None = None) -> None:
        """
        Update equity curve with new value.

        Args:
            equity: Current equity value
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self._equity_points.append(
            {
                "timestamp": timestamp,
                "equity": equity,
            }
        )

        # Maintain size limit
        if len(self._equity_points) > self._max_equity_points:
            self._equity_points = self._equity_points[-self._max_equity_points :]

        await self._check_and_update_metrics()

    async def add_trade(self, trade_data: dict[str, Any]) -> None:
        """
        Add completed trade to history.

        Args:
            trade_data: Trade information
        """
        self._trade_history.append(trade_data)

        # Maintain size limit
        if len(self._trade_history) > self._max_trade_history:
            self._trade_history = self._trade_history[-self._max_trade_history :]

        await self._check_and_update_metrics()

    async def add_signal(self, signal: Signal) -> None:
        """
        Add signal to history.

        Args:
            signal: Trading signal
        """
        self._signal_history.append(signal)

        # Maintain size limit
        if len(self._signal_history) > self._max_signal_history:
            self._signal_history = self._signal_history[-self._max_signal_history :]

        await self._check_and_update_metrics()

    async def _check_and_update_metrics(self) -> None:
        """Check if metrics need updating and update if necessary."""
        now = datetime.now(timezone.utc)
        if now - self._last_update >= self._update_interval:
            await self._update_metrics()
            self._last_update = now

    async def _update_metrics(self) -> None:
        """Update current metrics with latest data."""
        try:
            if not self._equity_points:
                return

            initial_capital = self._equity_points[0]["equity"]

            self._current_metrics = await self._calculator.calculate_comprehensive_metrics(
                equity_curve=self._equity_points,
                trades=self._trade_history,
                signals=self._signal_history,
                initial_capital=initial_capital,
            )

            self._logger.debug(f"Metrics updated for strategy {self.strategy_id}")

        except Exception as e:
            self._logger.error(f"Error updating metrics: {e}")

    def get_current_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.

        Returns:
            Current PerformanceMetrics
        """
        return self._current_metrics

    def get_metrics_summary(self) -> dict[str, Any]:
        """
        Get summary of key metrics.

        Returns:
            Dictionary with key performance indicators
        """
        metrics = self._current_metrics
        return {
            "strategy_id": self.strategy_id,
            "total_return": float(metrics.total_return),
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": float(metrics.max_drawdown),
            "win_rate": metrics.win_rate,
            "total_trades": metrics.total_trades,
            "signal_quality": metrics.signal_quality_score,
            "last_update": metrics.calculation_time.isoformat(),
        }

    async def reset_metrics(self) -> None:
        """Reset all metrics and data."""
        self._current_metrics = PerformanceMetrics()
        self._equity_points.clear()
        self._trade_history.clear()
        self._signal_history.clear()
        self._last_update = datetime.now(timezone.utc)

        self._logger.info(f"Metrics reset for strategy {self.strategy_id}")


class StrategyComparator:
    """
    Compare performance between multiple strategies.

    Provides tools for benchmarking strategies against each other
    and identifying best performers.
    """

    def __init__(self):
        """Initialize strategy comparator."""
        self._logger = get_logger(f"{__name__}.StrategyComparator")

    async def compare_strategies(
        self, strategy_metrics: dict[str, PerformanceMetrics]
    ) -> dict[str, Any]:
        """
        Compare multiple strategies.

        Args:
            strategy_metrics: Dictionary mapping strategy_id to PerformanceMetrics

        Returns:
            Comparison results
        """
        if len(strategy_metrics) < 2:
            return {"error": "Need at least 2 strategies to compare"}

        # Rank strategies by different metrics
        rankings = {}

        # Rank by Sharpe ratio
        sharpe_ranking = sorted(
            strategy_metrics.items(), key=lambda x: x[1].sharpe_ratio, reverse=True
        )
        rankings["sharpe_ratio"] = [
            {"strategy_id": sid, "value": metrics.sharpe_ratio} for sid, metrics in sharpe_ranking
        ]

        # Rank by total return
        return_ranking = sorted(
            strategy_metrics.items(), key=lambda x: x[1].total_return, reverse=True
        )
        rankings["total_return"] = [
            {"strategy_id": sid, "value": float(metrics.total_return)}
            for sid, metrics in return_ranking
        ]

        # Rank by max drawdown (lower is better)
        drawdown_ranking = sorted(
            strategy_metrics.items(), key=lambda x: x[1].max_drawdown, reverse=False
        )
        rankings["max_drawdown"] = [
            {"strategy_id": sid, "value": float(metrics.max_drawdown)}
            for sid, metrics in drawdown_ranking
        ]

        # Overall score (composite ranking)
        overall_scores = {}
        for strategy_id in strategy_metrics.keys():
            # Simple composite score
            metrics = strategy_metrics[strategy_id]
            score = (
                metrics.sharpe_ratio * 0.4
                + float(metrics.total_return) * 0.003  # Scale down return
                + (100 - float(metrics.max_drawdown)) * 0.01  # Invert drawdown
                + metrics.win_rate * 0.3
            )
            overall_scores[strategy_id] = score

        overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        rankings["overall_score"] = [
            {"strategy_id": sid, "value": score} for sid, score in overall_ranking
        ]

        return {
            "comparison_timestamp": datetime.now(timezone.utc).isoformat(),
            "strategies_compared": len(strategy_metrics),
            "rankings": rankings,
            "best_overall": overall_ranking[0][0] if overall_ranking else None,
        }
