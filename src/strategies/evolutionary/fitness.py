"""
Fitness Evaluation for Evolutionary Strategies.

This module provides fitness functions for evaluating trading strategy performance.
"""

import logging
from abc import ABC, abstractmethod

from src.backtesting.engine import BacktestResult


class FitnessFunction(ABC):
    """Abstract base class for fitness functions."""

    @abstractmethod
    def calculate(self, result: BacktestResult) -> float:
        """
        Calculate fitness score from backtest result.

        Args:
            result: Backtest result

        Returns:
            Fitness score
        """
        pass


class SharpeFitness(FitnessFunction):
    """Fitness based on Sharpe ratio."""

    def calculate(self, result: BacktestResult) -> float:
        """Calculate fitness based on Sharpe ratio."""
        return result.sharpe_ratio


class ReturnFitness(FitnessFunction):
    """Fitness based on total return."""

    def calculate(self, result: BacktestResult) -> float:
        """Calculate fitness based on total return."""
        return float(result.total_return)


class CompositeFitness(FitnessFunction):
    """Composite fitness combining multiple metrics."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        penalty_drawdown: bool = True,
        min_trades: int = 10,
    ):
        """
        Initialize composite fitness.

        Args:
            weights: Weights for different metrics
            penalty_drawdown: Whether to penalize high drawdown
            min_trades: Minimum trades required
        """
        self.weights = weights or {
            "sharpe_ratio": 0.4,
            "total_return": 0.3,
            "win_rate": 0.2,
            "profit_factor": 0.1,
        }
        self.penalty_drawdown = penalty_drawdown
        self.min_trades = min_trades

    def calculate(self, result: BacktestResult) -> float:
        """Calculate composite fitness score."""
        # Check minimum trades
        if result.total_trades < self.min_trades:
            return -1000.0  # Heavily penalize insufficient trades

        # Calculate weighted score
        score = 0.0

        # Sharpe ratio component
        if "sharpe_ratio" in self.weights:
            sharpe_component = result.sharpe_ratio * self.weights["sharpe_ratio"]
            score += sharpe_component

        # Return component (normalized)
        if "total_return" in self.weights:
            return_component = float(result.total_return) / 100 * self.weights["total_return"]
            score += return_component

        # Win rate component
        if "win_rate" in self.weights:
            win_rate_component = result.win_rate / 100 * self.weights["win_rate"]
            score += win_rate_component

        # Profit factor component (capped)
        if "profit_factor" in self.weights:
            profit_factor = min(result.profit_factor, 3.0)  # Cap at 3
            pf_component = profit_factor / 3.0 * self.weights["profit_factor"]
            score += pf_component

        # Apply drawdown penalty
        if self.penalty_drawdown:
            drawdown_penalty = float(result.max_drawdown) / 100
            score *= 1 - drawdown_penalty * 0.5  # Reduce score by up to 50%

        return score


class FitnessEvaluator:
    """
    Evaluates fitness of trading strategies.

    Supports multiple fitness functions and objectives.
    """

    def __init__(
        self,
        fitness_function: FitnessFunction | None = None,
        objectives: list[str] | None = None,
    ):
        """
        Initialize fitness evaluator.

        Args:
            fitness_function: Primary fitness function
            objectives: List of optimization objectives
        """
        self.fitness_function = fitness_function or CompositeFitness()
        self.objectives = objectives or ["sharpe_ratio", "total_return"]

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"FitnessEvaluator initialized with function {self.fitness_function.__class__.__name__} and objectives {self.objectives}"
        )

    def evaluate(self, result: BacktestResult) -> float:
        """
        Evaluate fitness of a backtest result.

        Args:
            result: Backtest result

        Returns:
            Fitness score
        """
        try:
            fitness = self.fitness_function.calculate(result)

            # Apply constraints
            fitness = self._apply_constraints(result, fitness)

            return fitness

        except Exception as e:
            self.logger.error(f"Fitness evaluation failed: {e!s}")
            return -float("inf")

    def evaluate_multi_objective(self, result: BacktestResult) -> dict[str, float]:
        """
        Evaluate multiple objectives.

        Args:
            result: Backtest result

        Returns:
            Dictionary of objective values
        """
        objectives = {}

        for objective in self.objectives:
            if objective == "sharpe_ratio":
                objectives[objective] = result.sharpe_ratio
            elif objective == "total_return":
                objectives[objective] = float(result.total_return)
            elif objective == "max_drawdown":
                objectives[objective] = -float(result.max_drawdown)  # Minimize
            elif objective == "win_rate":
                objectives[objective] = result.win_rate
            elif objective == "profit_factor":
                objectives[objective] = result.profit_factor
            elif objective == "volatility":
                objectives[objective] = -result.volatility  # Minimize
            elif objective == "sortino_ratio":
                objectives[objective] = result.sortino_ratio
            else:
                self.logger.warning(f"Unknown objective: {objective}")
                objectives[objective] = 0.0

        return objectives

    def _apply_constraints(self, result: BacktestResult, fitness: float) -> float:
        """Apply constraints to fitness score."""
        # Constraint: Maximum drawdown
        if float(result.max_drawdown) > 30:  # More than 30% drawdown
            fitness *= 0.5  # Halve fitness

        # Constraint: Minimum trades
        if result.total_trades < 10:
            fitness *= 0.1  # Severely penalize

        # Constraint: Minimum win rate
        if result.win_rate < 30:  # Less than 30% win rate
            fitness *= 0.7

        # Constraint: Negative return
        if float(result.total_return) < 0:
            fitness = min(fitness, 0)  # Cap at 0

        return fitness

    def compare(self, result1: BacktestResult, result2: BacktestResult) -> int:
        """
        Compare two backtest results.

        Args:
            result1: First result
            result2: Second result

        Returns:
            1 if result1 is better, -1 if result2 is better, 0 if equal
        """
        fitness1 = self.evaluate(result1)
        fitness2 = self.evaluate(result2)

        if fitness1 > fitness2:
            return 1
        elif fitness1 < fitness2:
            return -1
        else:
            return 0

    def rank(self, results: list[BacktestResult]) -> list[int]:
        """
        Rank backtest results by fitness.

        Args:
            results: List of backtest results

        Returns:
            List of indices sorted by fitness (best first)
        """
        fitnesses = [self.evaluate(r) for r in results]
        return sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)


class AdaptiveFitness(FitnessFunction):
    """Adaptive fitness that changes based on market conditions."""

    def __init__(self):
        """Initialize adaptive fitness."""
        self.market_regime = "normal"  # Could be: trending, volatile, ranging
        self.regime_weights = {
            "trending": {
                "total_return": 0.5,
                "sharpe_ratio": 0.3,
                "win_rate": 0.2,
            },
            "volatile": {
                "sharpe_ratio": 0.5,
                "max_drawdown": 0.3,
                "sortino_ratio": 0.2,
            },
            "ranging": {
                "win_rate": 0.4,
                "profit_factor": 0.3,
                "sharpe_ratio": 0.3,
            },
        }

    def set_market_regime(self, regime: str) -> None:
        """Set current market regime."""
        if regime in self.regime_weights:
            self.market_regime = regime
            logger = logging.getLogger(__name__)
            logger.info(f"Market regime set to: {regime}")

    def calculate(self, result: BacktestResult) -> float:
        """Calculate adaptive fitness based on market regime."""
        weights = self.regime_weights.get(self.market_regime, self.regime_weights["ranging"])

        score = 0.0

        for metric, weight in weights.items():
            if metric == "total_return":
                value = float(result.total_return) / 100
            elif metric == "sharpe_ratio":
                value = result.sharpe_ratio
            elif metric == "win_rate":
                value = result.win_rate / 100
            elif metric == "profit_factor":
                value = min(result.profit_factor, 3.0) / 3.0
            elif metric == "max_drawdown":
                value = 1 - float(result.max_drawdown) / 100
            elif metric == "sortino_ratio":
                value = result.sortino_ratio
            else:
                value = 0.0

            score += value * weight

        return score
