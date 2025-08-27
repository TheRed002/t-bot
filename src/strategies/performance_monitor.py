"""
Strategy Performance Monitor - Real-time performance tracking and analytics.

This module provides comprehensive performance monitoring, analytics, and reporting
for trading strategies including real-time metrics, risk analysis, and comparative
performance evaluation across multiple strategies.

Key Features:
- Real-time performance tracking and metrics calculation
- Risk-adjusted performance evaluation (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis and recovery tracking
- Strategy comparison and benchmarking
- Performance attribution analysis
- Automated alerting and reporting
- Historical performance persistence
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import numpy as np

from src.core.exceptions import PerformanceError
from src.core.types import (
    MarketRegime,
    Position,
    Trade,
)
from src.strategies.interfaces import (
    BaseStrategyInterface,
    MarketDataProviderInterface,
    StrategyDataRepositoryInterface,
)


class PerformanceMetrics:
    """Comprehensive performance metrics for a trading strategy."""

    def __init__(self, strategy_name: str):
        """Initialize performance metrics.

        Args:
            strategy_name: Name of the strategy
        """
        self.strategy_name = strategy_name

        # Basic metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.breakeven_trades = 0

        # P&L metrics
        self.total_pnl = Decimal("0")
        self.realized_pnl = Decimal("0")
        self.unrealized_pnl = Decimal("0")
        self.gross_profit = Decimal("0")
        self.gross_loss = Decimal("0")

        # Returns and ratios
        self.total_return = 0.0
        self.annualized_return = 0.0
        self.volatility = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        self.information_ratio = 0.0

        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.var_95 = 0.0
        self.var_99 = 0.0
        self.conditional_var_95 = 0.0
        self.beta = 1.0

        # Trade statistics
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.average_win = Decimal("0")
        self.average_loss = Decimal("0")
        self.largest_win = Decimal("0")
        self.largest_loss = Decimal("0")
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0

        # Time-based metrics
        self.average_holding_time = timedelta()
        self.total_time_in_market = timedelta()
        self.trades_per_day = 0.0
        self.strategy_start_time = datetime.now(timezone.utc)

        # Market exposure
        self.long_exposure_time = timedelta()
        self.short_exposure_time = timedelta()
        self.long_trades = 0
        self.short_trades = 0
        self.long_pnl = Decimal("0")
        self.short_pnl = Decimal("0")

        # Historical data
        self.daily_returns: list[float] = []
        self.monthly_returns: list[float] = []
        self.equity_curve: list[float] = []
        self.drawdown_curve: list[float] = []
        self.trade_history: list[Trade] = []

        # Benchmarking
        self.benchmark_returns: list[float] = []
        self.excess_returns: list[float] = []
        self.tracking_error = 0.0

        # Last update timestamp
        self.last_updated = datetime.now(timezone.utc)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for trading strategies.

    Tracks real-time performance, calculates risk-adjusted metrics, and provides
    detailed analytics for strategy evaluation and optimization.
    """

    def __init__(
        self,
        data_repository: StrategyDataRepositoryInterface | None = None,
        market_data_provider: MarketDataProviderInterface | None = None,
        update_interval_seconds: int = 60,
        calculation_window_days: int = 252,
    ):
        """Initialize performance monitor.

        Args:
            data_repository: Data repository for persistence
            update_interval_seconds: Frequency of metric updates
            calculation_window_days: Rolling window for calculations
        """
        self.data_repository = data_repository
        self.market_data_provider = market_data_provider
        self.update_interval = timedelta(seconds=update_interval_seconds)
        self.calculation_window = timedelta(days=calculation_window_days)

        # Strategy performance tracking
        self.strategy_metrics: dict[str, PerformanceMetrics] = {}
        self.monitored_strategies: dict[str, BaseStrategyInterface] = {}

        # Benchmark data
        self.benchmark_data: dict[str, list[float]] = {}
        self.market_regime = MarketRegime.UNKNOWN

        # Monitoring control
        self.monitoring_active = False
        self.monitoring_task: asyncio.Task | None = None
        self.last_update = datetime.now(timezone.utc)

        # Alert thresholds
        self.alert_thresholds = {
            "max_drawdown": 0.15,  # 15% maximum drawdown
            "min_sharpe_ratio": 0.5,  # Minimum acceptable Sharpe ratio
            "min_win_rate": 0.4,  # 40% minimum win rate
            "max_consecutive_losses": 10,  # Maximum consecutive losses
            "max_daily_loss": 0.05,  # 5% maximum daily loss
            "min_profit_factor": 1.1,  # Minimum profit factor
        }

        # Performance comparison
        self.strategy_rankings: dict[str, int] = {}
        self.performance_scores: dict[str, float] = {}

    async def add_strategy(self, strategy: BaseStrategyInterface) -> None:
        """
        Add a strategy to performance monitoring.

        Args:
            strategy: Strategy instance to monitor

        Raises:
            PerformanceError: If strategy is invalid or addition fails
        """
        try:
            # Validate input
            if not strategy:
                raise PerformanceError("Strategy cannot be None")

            if not hasattr(strategy, "name") or not strategy.name:
                raise PerformanceError("Strategy must have a valid name")

            strategy_name = strategy.name

            # Check if strategy already exists
            if strategy_name in self.strategy_metrics:
                raise PerformanceError(f"Strategy '{strategy_name}' is already being monitored")

            # Initialize metrics
            self.strategy_metrics[strategy_name] = PerformanceMetrics(strategy_name)
            self.monitored_strategies[strategy_name] = strategy

            # Load historical performance if available
            await self._load_historical_performance(strategy_name)

            # Start monitoring if not already active
            if not self.monitoring_active:
                await self.start_monitoring()

        except Exception as e:
            # Clean up on failure
            strategy_name = getattr(strategy, "name", "unknown") if strategy else "unknown"
            self.strategy_metrics.pop(strategy_name, None)
            self.monitored_strategies.pop(strategy_name, None)
            raise PerformanceError(f"Failed to add strategy '{strategy_name}': {e}") from e

    async def remove_strategy(self, strategy_name: str) -> None:
        """
        Remove a strategy from monitoring.

        Args:
            strategy_name: Name of strategy to remove
        """
        try:
            # Save final performance metrics
            await self._save_performance_metrics(strategy_name)

            # Remove from tracking
            self.strategy_metrics.pop(strategy_name, None)
            self.monitored_strategies.pop(strategy_name, None)
            self.strategy_rankings.pop(strategy_name, None)
            self.performance_scores.pop(strategy_name, None)

        except Exception as e:
            raise PerformanceError(f"Failed to remove strategy {strategy_name}: {e}") from e

    async def start_monitoring(self) -> None:
        """Start the performance monitoring loop."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Stop the performance monitoring loop."""
        self.monitoring_active = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that updates performance metrics."""
        while self.monitoring_active:
            try:
                # Update all strategy metrics
                await self._update_all_metrics()

                # Check for alerts
                await self._check_performance_alerts()

                # Update strategy rankings
                self._update_strategy_rankings()

                # Persist metrics to database
                await self._persist_metrics()

                # Wait for next update interval
                await asyncio.sleep(self.update_interval.total_seconds())

            except Exception:
                # Log error but continue monitoring
                continue

    async def _update_all_metrics(self) -> None:
        """Update performance metrics for all monitored strategies."""
        update_tasks = []

        for strategy_name in self.strategy_metrics.keys():
            task = self._update_strategy_metrics(strategy_name)
            update_tasks.append(task)

        # Update all strategies concurrently
        await asyncio.gather(*update_tasks, return_exceptions=True)

    async def _update_strategy_metrics(self, strategy_name: str) -> None:
        """Update performance metrics for a specific strategy."""
        try:
            if strategy_name not in self.monitored_strategies:
                return

            # Get strategy reference for potential future use
            # strategy = self.monitored_strategies[strategy_name]
            metrics = self.strategy_metrics[strategy_name]

            # Get current strategy state
            current_positions = await self._get_current_positions(strategy_name)
            recent_trades = await self._get_recent_trades(strategy_name)

            # Update trade statistics
            self._update_trade_statistics(metrics, recent_trades)

            # Update P&L metrics
            await self._update_pnl_metrics(metrics, current_positions, recent_trades)

            # Calculate risk-adjusted ratios
            self._calculate_risk_ratios(metrics)

            # Update drawdown analysis
            self._update_drawdown_analysis(metrics)

            # Calculate time-based metrics
            self._update_time_metrics(metrics, recent_trades)

            # Update market exposure metrics
            self._update_exposure_metrics(metrics, current_positions, recent_trades)

            # Calculate VaR and other risk metrics
            self._calculate_risk_metrics(metrics)

            # Mark as updated
            metrics.last_updated = datetime.now(timezone.utc)

        except Exception:
            # Log error but don't fail the entire update
            pass

    def _update_trade_statistics(self, metrics: PerformanceMetrics, trades: list[Trade]) -> None:
        """Update basic trade statistics."""
        if not trades:
            return

        # Count trade outcomes
        winning_count = sum(1 for trade in trades if trade.metadata.get("pnl", Decimal("0")) > 0)
        losing_count = sum(1 for trade in trades if trade.metadata.get("pnl", Decimal("0")) < 0)
        breakeven_count = sum(1 for trade in trades if trade.metadata.get("pnl", Decimal("0")) == 0)

        metrics.total_trades = len(trades)
        metrics.winning_trades = winning_count
        metrics.losing_trades = losing_count
        metrics.breakeven_trades = breakeven_count

        # Calculate win rate
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades

        # Calculate average wins/losses
        winning_trades = [trade for trade in trades if trade.metadata.get("pnl", Decimal("0")) > 0]
        losing_trades = [trade for trade in trades if trade.metadata.get("pnl", Decimal("0")) < 0]

        if winning_trades:
            metrics.average_win = Decimal(
                str(np.mean([float(t.metadata.get("pnl", Decimal("0"))) for t in winning_trades]))
            )
            metrics.largest_win = max(
                trade.metadata.get("pnl", Decimal("0")) for trade in winning_trades
            )

        if losing_trades:
            metrics.average_loss = Decimal(
                str(
                    abs(
                        np.mean([float(t.metadata.get("pnl", Decimal("0"))) for t in losing_trades])
                    )
                )
            )
            metrics.largest_loss = min(
                trade.metadata.get("pnl", Decimal("0")) for trade in losing_trades
            )

        # Calculate profit factor
        gross_profit = sum(trade.metadata.get("pnl", Decimal("0")) for trade in winning_trades)
        gross_loss = abs(sum(trade.metadata.get("pnl", Decimal("0")) for trade in losing_trades))

        metrics.gross_profit = gross_profit
        metrics.gross_loss = gross_loss

        if gross_loss > 0:
            metrics.profit_factor = float(gross_profit / gross_loss)

        # Calculate consecutive wins/losses
        self._calculate_consecutive_trades(metrics, trades)

    def _calculate_consecutive_trades(
        self, metrics: PerformanceMetrics, trades: list[Trade]
    ) -> None:
        """Calculate consecutive wins and losses statistics."""
        if not trades:
            return

        # Sort trades by timestamp
        sorted_trades = sorted(
            trades,
            key=lambda t: t.metadata.get("exit_time")
            or t.metadata.get("entry_time")
            or t.timestamp,
        )

        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0

        for trade in sorted_trades:
            pnl = trade.metadata.get("pnl", Decimal("0"))
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:  # Breakeven
                current_wins = 0
                current_losses = 0

        metrics.consecutive_wins = current_wins
        metrics.consecutive_losses = current_losses
        metrics.max_consecutive_wins = max_wins
        metrics.max_consecutive_losses = max_losses

    async def _update_pnl_metrics(
        self,
        metrics: PerformanceMetrics,
        positions: list[Position],
        trades: list[Trade],
    ) -> None:
        """Update P&L and return metrics."""
        # Calculate realized P&L from closed trades
        realized_pnl = sum(
            trade.metadata.get("pnl", Decimal("0"))
            for trade in trades
            if trade.metadata.get("is_closed", False)
        )
        metrics.realized_pnl = realized_pnl

        # Calculate unrealized P&L from open positions
        unrealized_pnl = Decimal("0")
        for position in positions:
            if position.is_open():  # Use Position model method instead of status attribute
                # Get current market price for position
                current_price = await self._get_current_price(position.symbol)
                if current_price:
                    position_pnl = self._calculate_position_pnl(position, current_price)
                    unrealized_pnl += position_pnl

        metrics.unrealized_pnl = unrealized_pnl
        metrics.total_pnl = realized_pnl + unrealized_pnl

        # Calculate returns based on initial capital
        if hasattr(metrics, "initial_capital") and metrics.initial_capital > 0:
            metrics.total_return = float(metrics.total_pnl / metrics.initial_capital)

        # Update equity curve
        current_equity = float(metrics.total_pnl)
        metrics.equity_curve.append(current_equity)

        # Keep only recent equity curve data
        max_points = int(self.calculation_window.days * 24)  # Hourly data points
        if len(metrics.equity_curve) > max_points:
            metrics.equity_curve = metrics.equity_curve[-max_points:]

    def _calculate_risk_ratios(self, metrics: PerformanceMetrics) -> None:
        """Calculate risk-adjusted performance ratios."""
        if len(metrics.daily_returns) < 30:  # Need at least 30 days of data
            return

        returns = np.array(metrics.daily_returns)

        # Calculate annualized metrics
        mean_return = np.mean(returns)
        volatility = np.std(returns)

        metrics.annualized_return = (1 + mean_return) ** 252 - 1
        metrics.volatility = volatility * np.sqrt(252)

        # Calculate Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        if volatility > 0:
            metrics.sharpe_ratio = (mean_return - risk_free_rate) / volatility * np.sqrt(252)

        # Calculate Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns)
            if downside_deviation > 0:
                metrics.sortino_ratio = (
                    (mean_return - risk_free_rate) / downside_deviation * np.sqrt(252)
                )

        # Calculate Calmar ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

        # Calculate Information ratio (vs benchmark)
        if len(metrics.benchmark_returns) == len(metrics.daily_returns):
            excess_returns = returns - np.array(metrics.benchmark_returns)
            tracking_error = np.std(excess_returns)
            if tracking_error > 0:
                metrics.information_ratio = np.mean(excess_returns) / tracking_error * np.sqrt(252)
                metrics.tracking_error = tracking_error * np.sqrt(252)

    def _update_drawdown_analysis(self, metrics: PerformanceMetrics) -> None:
        """Update drawdown analysis and recovery metrics."""
        if len(metrics.equity_curve) < 2:
            return

        equity_curve = np.array(metrics.equity_curve)

        # Calculate running maximum (peak)
        running_max = np.maximum.accumulate(equity_curve)

        # Calculate drawdown at each point
        drawdown = (equity_curve - running_max) / running_max

        # Update current drawdown
        metrics.current_drawdown = float(drawdown[-1])

        # Update maximum drawdown
        metrics.max_drawdown = float(np.min(drawdown))

        # Update drawdown curve
        metrics.drawdown_curve = drawdown.tolist()

        # Keep only recent drawdown data
        max_points = int(self.calculation_window.days * 24)
        if len(metrics.drawdown_curve) > max_points:
            metrics.drawdown_curve = metrics.drawdown_curve[-max_points:]

    def _update_time_metrics(self, metrics: PerformanceMetrics, trades: list[Trade]) -> None:
        """Update time-based performance metrics."""
        if not trades:
            return

        # Calculate average holding time
        completed_trades = [
            t for t in trades if t.metadata.get("exit_time") and t.metadata.get("entry_time")
        ]
        if completed_trades:
            holding_times = [
                t.metadata.get("exit_time") - t.metadata.get("entry_time") for t in completed_trades
            ]
            metrics.average_holding_time = sum(holding_times, timedelta()) / len(holding_times)

            # Total time in market
            metrics.total_time_in_market = sum(holding_times, timedelta())

        # Calculate trades per day
        strategy_runtime = datetime.now(timezone.utc) - metrics.strategy_start_time
        if strategy_runtime.days > 0:
            metrics.trades_per_day = metrics.total_trades / strategy_runtime.days

    def _update_exposure_metrics(
        self,
        metrics: PerformanceMetrics,
        positions: list[Position],
        trades: list[Trade],
    ) -> None:
        """Update market exposure and directional bias metrics."""
        # Count long vs short trades
        from src.core.types.trading import OrderSide

        long_trades = [t for t in trades if t.side == OrderSide.BUY]
        short_trades = [t for t in trades if t.side == OrderSide.SELL]

        metrics.long_trades = len(long_trades)
        metrics.short_trades = len(short_trades)

        # Calculate P&L by direction
        metrics.long_pnl = sum(t.metadata.get("pnl", Decimal("0")) for t in long_trades)
        metrics.short_pnl = sum(t.metadata.get("pnl", Decimal("0")) for t in short_trades)

        # Calculate exposure time (simplified - would need position history for accuracy)
        for position in positions:
            if position.is_open():  # Use Position model method instead of status attribute
                position_time = (
                    datetime.now(timezone.utc) - position.opened_at
                )  # Use opened_at instead of entry_time
                from src.core.types.trading import OrderSide

                if position.side == OrderSide.BUY:
                    metrics.long_exposure_time += position_time
                else:
                    metrics.short_exposure_time += position_time

    def _calculate_risk_metrics(self, metrics: PerformanceMetrics) -> None:
        """Calculate Value at Risk and other risk metrics."""
        if len(metrics.daily_returns) < 30:
            return

        returns = np.array(metrics.daily_returns)

        # Calculate VaR at different confidence levels
        metrics.var_95 = float(np.percentile(returns, 5))  # 95% VaR
        metrics.var_99 = float(np.percentile(returns, 1))  # 99% VaR

        # Calculate Conditional VaR (Expected Shortfall)
        var_95_threshold = metrics.var_95
        tail_losses = returns[returns <= var_95_threshold]
        if len(tail_losses) > 0:
            metrics.conditional_var_95 = float(np.mean(tail_losses))

        # Calculate Beta (vs benchmark if available)
        if len(metrics.benchmark_returns) == len(metrics.daily_returns):
            benchmark_returns = np.array(metrics.benchmark_returns)
            if np.std(benchmark_returns) > 0:
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                metrics.beta = covariance / benchmark_variance

    async def _check_performance_alerts(self) -> None:
        """Check for performance alerts and trigger notifications."""
        for strategy_name, metrics in self.strategy_metrics.items():
            alerts = []

            # Check drawdown alert
            if metrics.max_drawdown < -self.alert_thresholds["max_drawdown"]:
                alerts.append(f"Maximum drawdown exceeded: {metrics.max_drawdown:.2%}")

            # Check Sharpe ratio alert
            if metrics.sharpe_ratio < self.alert_thresholds["min_sharpe_ratio"]:
                alerts.append(f"Sharpe ratio below threshold: {metrics.sharpe_ratio:.2f}")

            # Check win rate alert
            if metrics.win_rate < self.alert_thresholds["min_win_rate"]:
                alerts.append(f"Win rate below threshold: {metrics.win_rate:.2%}")

            # Check consecutive losses
            if metrics.consecutive_losses > self.alert_thresholds["max_consecutive_losses"]:
                alerts.append(f"Consecutive losses: {metrics.consecutive_losses}")

            # Check profit factor
            if metrics.profit_factor < self.alert_thresholds["min_profit_factor"]:
                alerts.append(f"Profit factor below threshold: {metrics.profit_factor:.2f}")

            # Send alerts if any
            if alerts:
                await self._send_performance_alerts(strategy_name, alerts)

    def _update_strategy_rankings(self) -> None:
        """Update strategy rankings based on performance scores."""
        # Calculate composite performance score for each strategy
        for strategy_name, metrics in self.strategy_metrics.items():
            score = self._calculate_performance_score(metrics)
            self.performance_scores[strategy_name] = score

        # Rank strategies by performance score
        sorted_strategies = sorted(
            self.performance_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Update rankings
        for rank, (strategy_name, _) in enumerate(sorted_strategies, 1):
            self.strategy_rankings[strategy_name] = rank

    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate composite performance score."""
        try:
            # Weight different metrics
            weights = {
                "return": 0.25,
                "sharpe": 0.25,
                "drawdown": 0.20,
                "win_rate": 0.15,
                "profit_factor": 0.15,
            }

            # Normalize metrics to 0-1 scale
            return_score = max(0, min(1, (metrics.annualized_return + 0.2) / 0.4))  # -20% to 20%
            sharpe_score = max(0, min(1, metrics.sharpe_ratio / 3.0))  # 0 to 3
            drawdown_score = max(0, min(1, 1 + metrics.max_drawdown / 0.3))  # 0% to -30%
            win_rate_score = max(0, min(1, metrics.win_rate))  # 0% to 100%
            pf_score = max(0, min(1, (metrics.profit_factor - 0.5) / 2.5))  # 0.5 to 3.0

            # Calculate weighted score
            composite_score = (
                weights["return"] * return_score
                + weights["sharpe"] * sharpe_score
                + weights["drawdown"] * drawdown_score
                + weights["win_rate"] * win_rate_score
                + weights["profit_factor"] * pf_score
            )

            return composite_score

        except Exception:
            return 0.0

    async def get_strategy_performance(self, strategy_name: str) -> dict[str, Any]:
        """
        Get comprehensive performance report for a strategy.

        Args:
            strategy_name: Name of strategy

        Returns:
            Complete performance report

        Raises:
            PerformanceError: If strategy not found or invalid input
        """
        # Validate input
        if not strategy_name or not isinstance(strategy_name, str):
            raise PerformanceError("Invalid strategy_name: must be a non-empty string")

        if strategy_name not in self.strategy_metrics:
            raise PerformanceError(f"Strategy '{strategy_name}' not found in monitoring")

        metrics = self.strategy_metrics[strategy_name]

        return {
            "strategy_name": strategy_name,
            "last_updated": metrics.last_updated.isoformat(),
            "performance_rank": self.strategy_rankings.get(strategy_name, 0),
            "performance_score": self.performance_scores.get(strategy_name, 0.0),
            # Trade statistics
            "trade_stats": {
                "total_trades": metrics.total_trades,
                "winning_trades": metrics.winning_trades,
                "losing_trades": metrics.losing_trades,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "consecutive_wins": metrics.consecutive_wins,
                "consecutive_losses": metrics.consecutive_losses,
                "max_consecutive_wins": metrics.max_consecutive_wins,
                "max_consecutive_losses": metrics.max_consecutive_losses,
            },
            # P&L metrics
            "pnl_metrics": {
                "total_pnl": float(metrics.total_pnl),
                "realized_pnl": float(metrics.realized_pnl),
                "unrealized_pnl": float(metrics.unrealized_pnl),
                "gross_profit": float(metrics.gross_profit),
                "gross_loss": float(metrics.gross_loss),
                "average_win": float(metrics.average_win),
                "average_loss": float(metrics.average_loss),
                "largest_win": float(metrics.largest_win),
                "largest_loss": float(metrics.largest_loss),
            },
            # Risk-adjusted returns
            "risk_metrics": {
                "total_return": metrics.total_return,
                "annualized_return": metrics.annualized_return,
                "volatility": metrics.volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "calmar_ratio": metrics.calmar_ratio,
                "information_ratio": metrics.information_ratio,
                "tracking_error": metrics.tracking_error,
            },
            # Drawdown analysis
            "drawdown_metrics": {
                "max_drawdown": metrics.max_drawdown,
                "current_drawdown": metrics.current_drawdown,
                "drawdown_curve": metrics.drawdown_curve[-100:],  # Last 100 points
            },
            # Risk measures
            "risk_measures": {
                "var_95": metrics.var_95,
                "var_99": metrics.var_99,
                "conditional_var_95": metrics.conditional_var_95,
                "beta": metrics.beta,
            },
            # Time metrics
            "time_metrics": {
                "average_holding_time": str(metrics.average_holding_time),
                "trades_per_day": metrics.trades_per_day,
                "long_trades": metrics.long_trades,
                "short_trades": metrics.short_trades,
                "long_pnl": float(metrics.long_pnl),
                "short_pnl": float(metrics.short_pnl),
            },
            # Historical data
            "historical_data": {
                "equity_curve": metrics.equity_curve[-252:],  # Last year
                "daily_returns": metrics.daily_returns[-252:],  # Last year
                "monthly_returns": metrics.monthly_returns[-12:],  # Last 12 months
            },
        }

    async def get_comparative_analysis(self) -> dict[str, Any]:
        """
        Get comparative analysis across all strategies.

        Returns:
            Comparative performance analysis
        """
        if not self.strategy_metrics:
            return {}

        strategies = list(self.strategy_metrics.keys())

        # Compile comparison metrics
        comparison_data = {
            "strategy_rankings": self.strategy_rankings,
            "performance_scores": self.performance_scores,
            "total_strategies": len(strategies),
            "comparison_matrix": {},
        }

        # Create comparison matrix
        for strategy in strategies:
            metrics = self.strategy_metrics[strategy]
            comparison_data["comparison_matrix"][strategy] = {
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "volatility": metrics.volatility,
                "total_trades": metrics.total_trades,
            }

        # Calculate portfolio-level metrics if multiple strategies
        if len(strategies) > 1:
            portfolio_metrics = await self._calculate_portfolio_metrics()
            comparison_data["portfolio_metrics"] = portfolio_metrics

        return comparison_data

    async def _calculate_portfolio_metrics(self) -> dict[str, Any]:
        """Calculate portfolio-level performance metrics."""
        try:
            # Combine all strategy returns
            all_returns = []
            total_pnl = Decimal("0")
            total_trades = 0

            for metrics in self.strategy_metrics.values():
                if metrics.daily_returns:
                    all_returns.extend(metrics.daily_returns)
                total_pnl += metrics.total_pnl
                total_trades += metrics.total_trades

            if not all_returns:
                return {}

            returns = np.array(all_returns)

            # Calculate portfolio-level metrics
            portfolio_return = float(total_pnl)
            portfolio_volatility = np.std(returns) * np.sqrt(252)
            portfolio_sharpe = (
                np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            )

            # Calculate correlation matrix between strategies
            strategy_names = list(self.strategy_metrics.keys())
            correlation_matrix = {}

            for i, strategy1 in enumerate(strategy_names):
                correlation_matrix[strategy1] = {}
                for j, strategy2 in enumerate(strategy_names):
                    if i == j:
                        correlation_matrix[strategy1][strategy2] = 1.0
                    else:
                        returns1 = self.strategy_metrics[strategy1].daily_returns
                        returns2 = self.strategy_metrics[strategy2].daily_returns

                        if len(returns1) > 10 and len(returns2) > 10:
                            # Take minimum length for comparison
                            min_len = min(len(returns1), len(returns2))
                            corr = np.corrcoef(returns1[-min_len:], returns2[-min_len:])[0, 1]
                            correlation_matrix[strategy1][strategy2] = (
                                float(corr) if not np.isnan(corr) else 0.0
                            )
                        else:
                            correlation_matrix[strategy1][strategy2] = 0.0

            return {
                "total_return": portfolio_return,
                "volatility": portfolio_volatility,
                "sharpe_ratio": portfolio_sharpe,
                "total_trades": total_trades,
                "strategy_count": len(strategy_names),
                "correlation_matrix": correlation_matrix,
            }

        except Exception:
            return {}

    # Helper methods (implementations would depend on your data access patterns)
    async def _get_current_positions(self, strategy_name: str) -> list[Position]:
        """Get current positions for a strategy."""
        try:
            # Validate input
            if not strategy_name or not isinstance(strategy_name, str):
                raise ValueError("Invalid strategy_name parameter")

            if not self.data_repository:
                return []

            # Get positions from repository
            position_dicts = await self.data_repository.get_strategy_positions(strategy_name)

            if not position_dicts:
                return []

            # Convert dictionaries to Position objects
            positions = []
            for pos_dict in position_dicts:
                try:
                    # Validate required fields exist
                    required_fields = [
                        "symbol",
                        "side",
                        "quantity",
                        "entry_price",
                        "opened_at",
                        "exchange",
                    ]
                    if not all(field in pos_dict for field in required_fields):
                        continue

                    # Map from OrderSide enum values if needed
                    side_value = pos_dict.get("side")
                    if isinstance(side_value, str):
                        from src.core.types.trading import OrderSide

                        side = OrderSide.BUY if side_value.lower() == "buy" else OrderSide.SELL
                    else:
                        side = side_value

                    # Create Position object with correct field mapping
                    position = Position(
                        symbol=pos_dict.get("symbol"),
                        side=side,
                        quantity=Decimal(str(pos_dict.get("quantity", 0))),
                        entry_price=Decimal(str(pos_dict.get("entry_price", 0))),
                        current_price=(
                            Decimal(str(pos_dict.get("current_price", 0)))
                            if pos_dict.get("current_price")
                            else None
                        ),
                        unrealized_pnl=(
                            Decimal(str(pos_dict.get("unrealized_pnl", 0)))
                            if pos_dict.get("unrealized_pnl")
                            else None
                        ),
                        realized_pnl=Decimal(str(pos_dict.get("realized_pnl", 0))),
                        opened_at=pos_dict.get("opened_at"),
                        closed_at=pos_dict.get("closed_at"),
                        exchange=pos_dict.get("exchange"),
                        metadata=pos_dict.get("metadata", {}),
                    )
                    positions.append(position)
                except (ValueError, TypeError, KeyError) as e:
                    # Log specific error but continue processing
                    print(f"Error converting position data: {e}")
                    continue

            return positions

        except Exception as e:
            # Log error with more context
            print(f"Error getting positions for strategy '{strategy_name}': {e}")
            return []

    async def _get_recent_trades(
        self, strategy_name: str, limit: int = 1000, offset: int = 0
    ) -> list[Trade]:
        """Get recent trades for a strategy with pagination support."""
        try:
            # Validate input parameters
            self._validate_trade_query_params(strategy_name, limit, offset)

            if not self.data_repository:
                return []

            # Get trade dictionaries from repository
            trade_dicts = await self._fetch_trade_data(strategy_name)
            if not trade_dicts:
                return []

            # Apply pagination and convert to Trade objects
            paginated_trades = trade_dicts[offset : offset + limit]
            return self._convert_trade_dicts_to_objects(paginated_trades)

        except Exception as e:
            # Log error with more context
            print(f"Error getting trades for strategy '{strategy_name}': {e}")
            return []

    def _validate_trade_query_params(self, strategy_name: str, limit: int, offset: int) -> None:
        """Validate parameters for trade queries."""
        if not strategy_name or not isinstance(strategy_name, str):
            raise ValueError("Invalid strategy_name parameter")

        if limit <= 0 or limit > 10000:
            raise ValueError("Limit must be between 1 and 10000")

        if offset < 0:
            raise ValueError("Offset must be non-negative")

    async def _fetch_trade_data(self, strategy_name: str) -> list[dict[str, Any]]:
        """Fetch trade data from repository."""
        from datetime import timedelta

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=1)

        return await self.data_repository.get_strategy_trades(
            strategy_id=strategy_name, start_time=start_time, end_time=end_time
        )

    def _convert_trade_dicts_to_objects(self, trade_dicts: list[dict[str, Any]]) -> list[Trade]:
        """Convert trade dictionaries to Trade objects."""
        trades = []
        for trade_dict in trade_dicts:
            try:
                trade = self._create_trade_from_dict(trade_dict)
                if trade:
                    trades.append(trade)
            except (ValueError, TypeError, KeyError) as e:
                # Log specific error but continue processing
                print(f"Error converting trade data: {e}")
                continue

        return trades

    def _create_trade_from_dict(self, trade_dict: dict[str, Any]) -> Trade | None:
        """Create a Trade object from dictionary data."""
        # Validate and map required fields
        required_fields = [
            "trade_id",
            "order_id",
            "symbol",
            "side",
            "price",
            "quantity",
            "fee",
            "fee_currency",
            "timestamp",
            "exchange",
        ]

        if not all(field in trade_dict for field in required_fields):
            trade_dict = self._map_legacy_trade_fields(trade_dict)

        # Map OrderSide enum
        from src.core.types.trading import OrderSide

        side_value = trade_dict.get("side")
        side = (
            OrderSide.BUY
            if isinstance(side_value, str) and side_value.lower() == "buy"
            else (
                OrderSide.SELL
                if isinstance(side_value, str) and side_value.lower() == "sell"
                else side_value
            )
        )

        # Create Trade object
        trade = Trade(
            trade_id=trade_dict.get("trade_id", str(trade_dict.get("id", ""))),
            order_id=trade_dict.get("order_id", ""),
            symbol=trade_dict.get("symbol", ""),
            side=side,
            price=Decimal(str(trade_dict.get("price", 0))),
            quantity=Decimal(str(trade_dict.get("quantity", 0))),
            fee=Decimal(str(trade_dict.get("fee", 0))),
            fee_currency=trade_dict.get("fee_currency", "USD"),
            timestamp=trade_dict.get("timestamp")
            or trade_dict.get("entry_time")
            or datetime.now(timezone.utc),
            exchange=trade_dict.get("exchange", "unknown"),
            is_maker=trade_dict.get("is_maker", False),
            metadata=trade_dict.get("metadata", {}),
        )

        # Store extended fields in metadata for backwards compatibility
        extended_fields = {}
        if "pnl" in trade_dict:
            extended_fields["pnl"] = Decimal(str(trade_dict.get("pnl", 0)))
        if "entry_time" in trade_dict:
            extended_fields["entry_time"] = trade_dict.get("entry_time")
        if "exit_time" in trade_dict:
            extended_fields["exit_time"] = trade_dict.get("exit_time")
        if "is_closed" in trade_dict:
            extended_fields["is_closed"] = trade_dict.get("is_closed", False)

        # Merge with existing metadata
        trade.metadata.update(extended_fields)

        return trade

    def _map_legacy_trade_fields(self, trade_dict: dict[str, Any]) -> dict[str, Any]:
        """Map legacy field names to current Trade model fields."""
        mapped_dict = trade_dict.copy()

        # Legacy field mappings
        field_mappings = {
            "id": "trade_id",
            "entry_time": "timestamp",
            "created_at": "timestamp",
            "executed_at": "timestamp",
        }

        for old_field, new_field in field_mappings.items():
            if old_field in mapped_dict and new_field not in mapped_dict:
                mapped_dict[new_field] = mapped_dict[old_field]

        # Set default values for required fields
        defaults = {
            "trade_id": mapped_dict.get("id", "unknown"),
            "order_id": mapped_dict.get("order_id", "unknown"),
            "fee_currency": "USD",
            "exchange": "unknown",
            "timestamp": mapped_dict.get("entry_time") or datetime.now(timezone.utc),
        }

        for field, default_value in defaults.items():
            if field not in mapped_dict:
                mapped_dict[field] = default_value

        return mapped_dict

    async def _get_current_price(self, symbol: str) -> Decimal | None:
        """Get current market price for a symbol."""
        try:
            if not self.market_data_provider:
                return None

            return await self.market_data_provider.get_current_price(symbol)

        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return None

    def _calculate_position_pnl(self, position: Position, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L for a position."""
        # Simple P&L calculation - enhance based on your position model
        price_diff = current_price - position.entry_price
        if position.side == "buy":
            return position.quantity * price_diff
        else:
            return position.quantity * (position.entry_price - current_price)

    async def _load_historical_performance(self, strategy_name: str) -> None:
        """Load historical performance data from database."""
        try:
            if not self.data_repository:
                return

            # Load historical metrics
            history = await self.data_repository.load_performance_history(strategy_id=strategy_name)

            if history and strategy_name in self.strategy_metrics:
                metrics = self.strategy_metrics[strategy_name]
                # Process historical data to populate metrics
                for hist_data in history:
                    if "daily_returns" in hist_data:
                        metrics.daily_returns.extend(hist_data["daily_returns"])
                    if "equity_curve" in hist_data:
                        metrics.equity_curve.extend(hist_data["equity_curve"])

        except Exception as e:
            # Log error but don't fail initialization
            print(f"Error loading historical performance for {strategy_name}: {e}")

    async def _save_performance_metrics(self, strategy_name: str) -> None:
        """Save performance metrics to database."""
        try:
            if not self.data_repository or strategy_name not in self.strategy_metrics:
                return

            metrics = self.strategy_metrics[strategy_name]

            # Convert metrics to dictionary for storage
            metrics_dict = {
                "total_trades": metrics.total_trades,
                "winning_trades": metrics.winning_trades,
                "losing_trades": metrics.losing_trades,
                "total_pnl": float(metrics.total_pnl),
                "realized_pnl": float(metrics.realized_pnl),
                "unrealized_pnl": float(metrics.unrealized_pnl),
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "daily_returns": metrics.daily_returns[-30:],  # Last 30 days
                "equity_curve": metrics.equity_curve[-100:],  # Last 100 points
            }

            await self.data_repository.save_performance_metrics(
                strategy_id=strategy_name, metrics=metrics_dict, timestamp=metrics.last_updated
            )

        except Exception as e:
            # Log error but continue operation
            print(f"Error saving performance metrics for {strategy_name}: {e}")

    async def _persist_metrics(self) -> None:
        """Persist all metrics to database."""
        if not self.data_repository:
            return

        # Save metrics for all strategies
        persist_tasks = []
        for strategy_name in self.strategy_metrics:
            task = self._save_performance_metrics(strategy_name)
            persist_tasks.append(task)

        # Execute all saves concurrently
        try:
            await asyncio.gather(*persist_tasks, return_exceptions=True)
        except Exception as e:
            # Log error but don't fail monitoring loop
            print(f"Error persisting metrics: {e}")

    async def _send_performance_alerts(self, strategy_name: str, alerts: list[str]) -> None:
        """Send performance alerts."""
        # Implementation depends on your alerting system
        pass
