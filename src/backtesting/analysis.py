"""
Advanced Analysis Methods for Backtesting.

This module provides Monte Carlo simulations and Walk-Forward analysis
for robust strategy evaluation and parameter optimization.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import BacktestError
from src.core.logging import get_logger
from src.utils.decorators import time_execution

from .engine import BacktestConfig, BacktestEngine

logger = get_logger(__name__)


class MonteCarloAnalyzer:
    """
    Monte Carlo simulation for backtesting robustness analysis.

    Provides:
    - Random trade sequence permutation
    - Bootstrap resampling
    - Confidence interval estimation
    - Risk assessment through simulation
    """

    def __init__(
        self,
        num_simulations: int = 1000,
        confidence_level: float = 0.95,
        seed: int | None = None,
    ):
        """
        Initialize Monte Carlo analyzer.

        Args:
            num_simulations: Number of simulation runs
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            seed: Random seed for reproducibility
        """
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self.seed = seed

        if seed:
            np.random.seed(seed)

        logger.info(
            "MonteCarloAnalyzer initialized",
            simulations=num_simulations,
            confidence=confidence_level,
        )

    @time_execution
    async def analyze_trades(
        self, trades: list[dict[str, Any]], initial_capital: float
    ) -> dict[str, Any]:
        """
        Perform Monte Carlo analysis on trade sequences.

        Args:
            trades: List of historical trades
            initial_capital: Starting capital

        Returns:
            Analysis results including confidence intervals
        """
        if not trades:
            raise BacktestError("No trades to analyze")

        logger.info("Starting Monte Carlo analysis", num_trades=len(trades))

        # Extract trade returns
        trade_returns = [t["pnl"] / initial_capital for t in trades]

        # Run simulations
        simulation_results = []

        for _i in range(self.num_simulations):
            # Resample trades with replacement
            resampled_indices = np.random.choice(
                len(trade_returns), size=len(trade_returns), replace=True
            )
            resampled_returns = [trade_returns[idx] for idx in resampled_indices]

            # Calculate cumulative return
            cumulative_return = np.prod([1 + r for r in resampled_returns]) - 1

            # Calculate metrics for this simulation
            sim_metrics = self._calculate_simulation_metrics(resampled_returns, initial_capital)
            sim_metrics["cumulative_return"] = cumulative_return
            simulation_results.append(sim_metrics)

        # Analyze results
        analysis = self._analyze_simulation_results(simulation_results)

        logger.info("Monte Carlo analysis completed", simulations_run=len(simulation_results))
        return analysis

    @time_execution
    async def analyze_returns(
        self, daily_returns: list[float], num_days: int = 252
    ) -> dict[str, Any]:
        """
        Perform Monte Carlo simulation on return paths.

        Args:
            daily_returns: Historical daily returns
            num_days: Number of days to simulate forward

        Returns:
            Simulation results with probable outcomes
        """
        if not daily_returns:
            raise BacktestError("No returns to analyze")

        returns_array = np.array(daily_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        logger.info(
            "Simulating return paths",
            mean_return=mean_return,
            volatility=std_return,
        )

        # Generate random return paths
        paths = []

        for _ in range(self.num_simulations):
            # Generate random returns using historical parameters
            simulated_returns = np.random.normal(mean_return, std_return, num_days)

            # Calculate cumulative path
            path = np.cumprod(1 + simulated_returns)
            paths.append(path)

        paths_array = np.array(paths)

        # Calculate statistics
        final_values = paths_array[:, -1]

        # Percentiles
        lower_percentile = (1 - self.confidence_level) / 2 * 100
        upper_percentile = (1 + self.confidence_level) / 2 * 100

        results = {
            "expected_return": float(np.mean(final_values) - 1),
            "median_return": float(np.median(final_values) - 1),
            "std_deviation": float(np.std(final_values)),
            "confidence_interval": {
                "lower": float(np.percentile(final_values, lower_percentile) - 1),
                "upper": float(np.percentile(final_values, upper_percentile) - 1),
            },
            "probability_profit": float(np.mean(final_values > 1)),
            "max_simulated_return": float(np.max(final_values) - 1),
            "min_simulated_return": float(np.min(final_values) - 1),
            "percentiles": {
                "5th": float(np.percentile(final_values, 5) - 1),
                "25th": float(np.percentile(final_values, 25) - 1),
                "75th": float(np.percentile(final_values, 75) - 1),
                "95th": float(np.percentile(final_values, 95) - 1),
            },
        }

        return results

    def _calculate_simulation_metrics(
        self, returns: list[float], initial_capital: float
    ) -> dict[str, Any]:
        """Calculate metrics for a single simulation."""
        if not returns:
            return {}

        # Calculate equity curve
        equity = initial_capital
        equity_curve = [equity]

        for ret in returns:
            equity *= 1 + ret
            equity_curve.append(equity)

        # Calculate drawdown
        peak = initial_capital
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        # Calculate Sharpe ratio
        returns_array = np.array(returns)
        if len(returns_array) > 1:
            sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        else:
            sharpe = 0

        return {
            "final_equity": equity,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "volatility": np.std(returns_array) if len(returns_array) > 1 else 0,
        }

    def _analyze_simulation_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze aggregated simulation results."""
        # Extract metrics
        returns = [r["cumulative_return"] for r in results]
        drawdowns = [r["max_drawdown"] for r in results]
        sharpes = [r["sharpe_ratio"] for r in results]

        # Calculate percentiles
        lower_percentile = (1 - self.confidence_level) / 2 * 100
        upper_percentile = (1 + self.confidence_level) / 2 * 100

        analysis = {
            "return_statistics": {
                "mean": float(np.mean(returns)),
                "median": float(np.median(returns)),
                "std": float(np.std(returns)),
                "confidence_interval": {
                    "lower": float(np.percentile(returns, lower_percentile)),
                    "upper": float(np.percentile(returns, upper_percentile)),
                },
            },
            "drawdown_statistics": {
                "mean": float(np.mean(drawdowns)),
                "median": float(np.median(drawdowns)),
                "worst_case": float(np.max(drawdowns)),
                "confidence_interval": {
                    "lower": float(np.percentile(drawdowns, lower_percentile)),
                    "upper": float(np.percentile(drawdowns, upper_percentile)),
                },
            },
            "sharpe_statistics": {
                "mean": float(np.mean(sharpes)),
                "median": float(np.median(sharpes)),
                "confidence_interval": {
                    "lower": float(np.percentile(sharpes, lower_percentile)),
                    "upper": float(np.percentile(sharpes, upper_percentile)),
                },
            },
            "risk_metrics": {
                "probability_loss": float(np.mean([r < 0 for r in returns])),
                "expected_shortfall": float(
                    np.mean([r for r in returns if r < np.percentile(returns, 5)])
                ),
                "tail_ratio": float(
                    np.percentile(returns, 95) / abs(np.percentile(returns, 5))
                    if np.percentile(returns, 5) != 0
                    else 0
                ),
            },
        }

        return analysis


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis for robust parameter optimization.

    Provides:
    - Out-of-sample testing
    - Rolling window optimization
    - Parameter stability analysis
    - Performance degradation detection
    """

    def __init__(
        self,
        optimization_window: int = 252,  # Trading days
        test_window: int = 63,  # Trading days
        step_size: int | None = None,  # If None, equals test_window
    ):
        """
        Initialize walk-forward analyzer.

        Args:
            optimization_window: Days for in-sample optimization
            test_window: Days for out-of-sample testing
            step_size: Days to step forward (if None, non-overlapping windows)
        """
        self.optimization_window = optimization_window
        self.test_window = test_window
        self.step_size = step_size or test_window

        logger.info(
            "WalkForwardAnalyzer initialized",
            opt_window=optimization_window,
            test_window=test_window,
            step=self.step_size,
        )

    @time_execution
    async def analyze(
        self,
        strategy_class: type,
        parameter_ranges: dict[str, list[Any]],
        market_data: pd.DataFrame,
        optimization_metric: str = "sharpe_ratio",
    ) -> dict[str, Any]:
        """
        Perform walk-forward analysis.

        Args:
            strategy_class: Strategy class to optimize
            parameter_ranges: Parameter ranges for optimization
            market_data: Historical market data
            optimization_metric: Metric to optimize

        Returns:
            Walk-forward analysis results
        """
        logger.info(
            "Starting walk-forward analysis",
            strategy=strategy_class.__name__,
            parameters=list(parameter_ranges.keys()),
        )

        # Generate windows
        windows = self._generate_windows(market_data)

        if not windows:
            raise BacktestError("Insufficient data for walk-forward analysis")

        # Run analysis for each window
        window_results = []

        for i, (opt_start, opt_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Processing window {i + 1}/{len(windows)}")

            # Optimize on in-sample data
            opt_data = market_data[opt_start:opt_end]
            best_params = await self._optimize_parameters(
                strategy_class, parameter_ranges, opt_data, optimization_metric
            )

            # Test on out-of-sample data
            test_data = market_data[test_start:test_end]
            test_result = await self._test_parameters(strategy_class, best_params, test_data)

            window_results.append(
                {
                    "window": i,
                    "optimization_period": (opt_start, opt_end),
                    "test_period": (test_start, test_end),
                    "best_parameters": best_params,
                    "in_sample_performance": best_params.get("performance", {}),
                    "out_of_sample_performance": test_result,
                }
            )

        # Analyze results
        analysis = self._analyze_results(window_results, optimization_metric)

        logger.info("Walk-forward analysis completed", windows_analyzed=len(windows))
        return analysis

    def _generate_windows(
        self, data: pd.DataFrame
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """Generate optimization and test windows."""
        windows = []

        total_days = len(data)
        min_required = self.optimization_window + self.test_window

        if total_days < min_required:
            return windows

        current_pos = 0

        while current_pos + min_required <= total_days:
            opt_start = data.index[current_pos]
            opt_end = data.index[current_pos + self.optimization_window - 1]
            test_start = data.index[current_pos + self.optimization_window]
            test_end = data.index[min(current_pos + min_required - 1, total_days - 1)]

            windows.append((opt_start, opt_end, test_start, test_end))
            current_pos += self.step_size

        return windows

    async def _optimize_parameters(
        self,
        strategy_class: type,
        parameter_ranges: dict[str, list[Any]],
        data: pd.DataFrame,
        metric: str,
    ) -> dict[str, Any]:
        """Optimize parameters on given data."""
        best_params = None
        best_score = -float("inf")

        # Generate parameter combinations
        from itertools import product

        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())

        for combination in product(*param_values):
            params = dict(zip(param_names, combination, strict=False))

            # Run backtest with these parameters
            strategy = strategy_class(**params)
            config = BacktestConfig(
                start_date=data.index[0],
                end_date=data.index[-1],
                symbols=["TEST"],  # Placeholder
            )

            engine = BacktestEngine(config, strategy)
            result = await engine.run()

            # Get optimization metric
            score = getattr(result, metric, 0)

            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_params["performance"] = {
                    metric: score,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": float(result.max_drawdown),
                }

        return best_params

    async def _test_parameters(
        self, strategy_class: type, parameters: dict[str, Any], data: pd.DataFrame
    ) -> dict[str, Any]:
        """Test parameters on out-of-sample data."""
        # Remove performance data from parameters
        params = {k: v for k, v in parameters.items() if k != "performance"}

        # Run backtest
        strategy = strategy_class(**params)
        config = BacktestConfig(
            start_date=data.index[0],
            end_date=data.index[-1],
            symbols=["TEST"],
        )

        engine = BacktestEngine(config, strategy)
        result = await engine.run()

        return {
            "total_return": float(result.total_return),
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": float(result.max_drawdown),
            "win_rate": result.win_rate,
        }

    def _analyze_results(
        self, window_results: list[dict[str, Any]], optimization_metric: str
    ) -> dict[str, Any]:
        """Analyze walk-forward results."""
        # Extract performance metrics
        in_sample_scores = [
            w["in_sample_performance"].get(optimization_metric, 0) for w in window_results
        ]
        out_sample_scores = [
            w["out_of_sample_performance"].get(
                optimization_metric.replace("_", " ").title().replace(" ", "_").lower(), 0
            )
            for w in window_results
        ]

        # Calculate efficiency ratio
        if in_sample_scores and out_sample_scores:
            efficiency_ratio = np.mean(out_sample_scores) / np.mean(in_sample_scores)
        else:
            efficiency_ratio = 0

        # Parameter stability analysis
        parameter_stability = self._analyze_parameter_stability(window_results)

        return {
            "summary": {
                "windows_analyzed": len(window_results),
                "avg_in_sample_performance": float(np.mean(in_sample_scores)),
                "avg_out_sample_performance": float(np.mean(out_sample_scores)),
                "efficiency_ratio": float(efficiency_ratio),
                "performance_degradation": float(
                    (np.mean(in_sample_scores) - np.mean(out_sample_scores))
                    / np.mean(in_sample_scores)
                    * 100
                    if np.mean(in_sample_scores) > 0
                    else 0
                ),
            },
            "parameter_stability": parameter_stability,
            "window_details": window_results,
        }

    def _analyze_parameter_stability(self, window_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze how stable parameters are across windows."""
        if not window_results:
            return {}

        # Extract all parameter names
        all_params = set()
        for w in window_results:
            if "best_parameters" in w:
                all_params.update(w["best_parameters"].keys())

        all_params.discard("performance")  # Remove non-parameter key

        stability = {}

        for param in all_params:
            values = [
                w["best_parameters"].get(param)
                for w in window_results
                if param in w.get("best_parameters", {})
            ]

            if values:
                # Calculate stability metrics
                unique_values = list(set(values))

                if all(isinstance(v, int | float) for v in values):
                    stability[param] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "cv": (
                            float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
                        ),
                    }
                else:
                    # For non-numeric parameters
                    from collections import Counter

                    counts = Counter(values)
                    most_common = counts.most_common(1)[0]

                    stability[param] = {
                        "most_common": most_common[0],
                        "frequency": most_common[1] / len(values),
                        "unique_values": len(unique_values),
                    }

        return stability
