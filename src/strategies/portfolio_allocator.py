"""
Strategy Portfolio Allocator - Dynamic strategy allocation and switching.

This module implements a sophisticated portfolio allocation system that dynamically
allocates capital across multiple trading strategies based on their performance,
risk metrics, and market conditions.

Key Features:
- Dynamic capital allocation based on strategy performance
- Risk-adjusted performance evaluation (Sharpe ratio, Sortino ratio)
- Strategy correlation analysis for diversification
- Market regime-aware allocation adjustments
- Real-time strategy switching based on performance deterioration
- Portfolio-level risk management and limits
- Performance attribution and analytics
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import numpy as np
from scipy.optimize import minimize

from src.core.exceptions import AllocationError
from src.core.types import (
    MarketRegime,
    Signal,
    SignalDirection,
    StrategyStatus,
    StrategyType,
)
from src.risk_management.base import BaseRiskManager
from src.strategies.interfaces import BaseStrategyInterface
from src.utils.decorators import time_execution


class StrategyAllocation:
    """Represents allocation for a single strategy."""

    def __init__(
        self,
        strategy: BaseStrategyInterface,
        target_weight: float,
        current_weight: float,
        allocated_capital: Decimal,
        max_allocation: float = 0.4,
        min_allocation: float = 0.05,
    ):
        """Initialize strategy allocation.

        Args:
            strategy: Strategy instance
            target_weight: Target portfolio weight (0-1)
            current_weight: Current portfolio weight (0-1)
            allocated_capital: Currently allocated capital
            max_allocation: Maximum allowed allocation
            min_allocation: Minimum required allocation
        """
        self.strategy = strategy
        self.target_weight = target_weight
        self.current_weight = current_weight
        self.allocated_capital = allocated_capital
        self.max_allocation = max_allocation
        self.min_allocation = min_allocation

        # Performance tracking
        self.daily_returns: list[float] = []
        self.cumulative_pnl = Decimal("0")
        self.trade_count = 0
        self.win_rate = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.max_drawdown = 0.0
        self.volatility = 0.0

        # Risk metrics
        self.var_95 = 0.0  # Value at Risk (95% confidence)
        self.correlation_with_market = 0.0
        self.beta = 1.0

        # Allocation metadata
        self.last_rebalance = datetime.now(timezone.utc)
        self.rebalance_threshold = 0.05  # 5% weight deviation triggers rebalance
        self.performance_lookback_days = 30


class PortfolioAllocator:
    """
    Dynamic portfolio allocator for trading strategies.

    Manages capital allocation across multiple strategies using modern portfolio
    theory principles, performance-based allocation, and risk management.
    """

    def __init__(
        self,
        total_capital: Decimal,
        risk_manager: BaseRiskManager,
        max_strategies: int = 10,
        rebalance_frequency_hours: int = 24,
        min_strategy_allocation: float = 0.05,
        max_strategy_allocation: float = 0.4,
    ):
        """Initialize portfolio allocator.

        Args:
            total_capital: Total portfolio capital
            risk_manager: Risk management system
            max_strategies: Maximum number of active strategies
            rebalance_frequency_hours: Hours between rebalancing
            min_strategy_allocation: Minimum allocation per strategy
            max_strategy_allocation: Maximum allocation per strategy
        """
        self.total_capital = total_capital
        self.risk_manager = risk_manager
        self.max_strategies = max_strategies
        self.rebalance_frequency = timedelta(hours=rebalance_frequency_hours)
        self.min_strategy_allocation = min_strategy_allocation
        self.max_strategy_allocation = max_strategy_allocation

        # Strategy allocations
        self.allocations: dict[str, StrategyAllocation] = {}
        self.strategy_queue: list[BaseStrategyInterface] = []  # Strategies waiting for allocation

        # Portfolio state
        self.last_rebalance = datetime.now(timezone.utc)
        self.portfolio_value = total_capital
        self.available_capital = total_capital
        self.allocated_capital = Decimal("0")

        # Performance tracking
        self.portfolio_returns: list[float] = []
        self.benchmark_returns: list[float] = []  # For comparison
        self.portfolio_metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }

        # Risk parameters
        self.max_portfolio_leverage = 1.0
        self.max_correlation_threshold = 0.7  # Maximum correlation between strategies
        self.volatility_target = 0.15  # Target portfolio volatility (15%)
        self.max_drawdown_limit = 0.20  # Stop if drawdown exceeds 20%

        # Market regime tracking
        self.current_regime = MarketRegime.NORMAL
        self.regime_allocations = {
            MarketRegime.BULL: {"risk_multiplier": 1.2, "diversification_weight": 0.8},
            MarketRegime.BEAR: {"risk_multiplier": 0.7, "diversification_weight": 1.3},
            MarketRegime.SIDEWAYS: {"risk_multiplier": 1.0, "diversification_weight": 1.0},
            MarketRegime.HIGH_VOLATILITY: {"risk_multiplier": 0.6, "diversification_weight": 1.5},
            MarketRegime.LOW_VOLATILITY: {"risk_multiplier": 1.1, "diversification_weight": 0.9},
        }

    @time_execution
    async def add_strategy(
        self, strategy: BaseStrategyInterface, initial_weight: float = 0.1
    ) -> bool:
        """
        Add a new strategy to the portfolio.

        Args:
            strategy: Strategy to add
            initial_weight: Initial portfolio weight

        Returns:
            True if strategy was added successfully
        """
        try:
            # Validate strategy
            if not await self._validate_strategy(strategy):
                return False

            # Check if we have capacity
            if len(self.allocations) >= self.max_strategies:
                # Add to queue for later consideration
                self.strategy_queue.append(strategy)
                return True

            # Calculate initial allocation
            initial_capital = self.total_capital * Decimal(str(initial_weight))

            # Create allocation
            allocation = StrategyAllocation(
                strategy=strategy,
                target_weight=initial_weight,
                current_weight=0.0,
                allocated_capital=Decimal("0"),
                max_allocation=self.max_strategy_allocation,
                min_allocation=self.min_strategy_allocation,
            )

            # Add to allocations
            self.allocations[strategy.name] = allocation

            # Start the strategy
            await strategy.start()

            # Trigger immediate rebalancing to allocate capital
            await self.rebalance_portfolio()

            return True

        except Exception as e:
            raise AllocationError(f"Failed to add strategy {strategy.name}: {e}")

    async def _validate_strategy(self, strategy: BaseStrategyInterface) -> bool:
        """
        Validate that a strategy meets portfolio requirements.

        Args:
            strategy: Strategy to validate

        Returns:
            True if strategy is valid for inclusion
        """
        try:
            # Check strategy status
            if strategy.status not in [StrategyStatus.STOPPED, StrategyStatus.RUNNING]:
                return False

            # Check for duplicate names
            if strategy.name in self.allocations:
                return False

            # Check correlation with existing strategies
            if len(self.allocations) > 0:
                correlation = await self._calculate_strategy_correlation(strategy)
                if correlation > self.max_correlation_threshold:
                    return False

            # Check strategy requirements
            if hasattr(strategy, "validate_signal"):
                # Test signal validation
                test_signal = Signal(
                    direction=SignalDirection.BUY,
                    strength=0.8,
                    timestamp=datetime.now(timezone.utc),
                    symbol="BTCUSDT",
                    source=strategy.name,
                    metadata={},
                )
                if not await strategy.validate_signal(test_signal):
                    return False

            return True

        except Exception:
            return False

    async def _calculate_strategy_correlation(self, new_strategy: BaseStrategyInterface) -> float:
        """
        Calculate correlation between new strategy and existing portfolio.

        Args:
            new_strategy: Strategy to calculate correlation for

        Returns:
            Maximum correlation with existing strategies
        """
        try:
            # For new strategies, estimate correlation based on strategy type
            # This is a simplified approach - in production, you'd use historical returns

            strategy_type_correlations = {
                (StrategyType.MOMENTUM, StrategyType.TREND_FOLLOWING): 0.8,
                (StrategyType.MEAN_REVERSION, StrategyType.ARBITRAGE): 0.3,
                (StrategyType.MARKET_MAKING, StrategyType.ARBITRAGE): 0.4,
                (StrategyType.VOLATILITY_BREAKOUT, StrategyType.BREAKOUT): 0.7,
                (StrategyType.ENSEMBLE, StrategyType.HYBRID): 0.9,
            }

            max_correlation = 0.0
            for allocation in self.allocations.values():
                existing_type = allocation.strategy.strategy_type
                new_type = new_strategy.strategy_type

                # Check direct correlation
                correlation = strategy_type_correlations.get((existing_type, new_type), 0.2)
                correlation = max(
                    correlation, strategy_type_correlations.get((new_type, existing_type), 0.2)
                )

                max_correlation = max(max_correlation, correlation)

            return max_correlation

        except Exception:
            return 1.0  # Conservative estimate

    @time_execution
    async def rebalance_portfolio(self) -> dict[str, Any]:
        """
        Rebalance portfolio allocations based on performance and risk metrics.

        Returns:
            Rebalancing results and metrics
        """
        try:
            # Update strategy performance metrics
            await self._update_strategy_metrics()

            # Calculate optimal weights
            optimal_weights = await self._calculate_optimal_weights()

            # Apply regime-based adjustments
            adjusted_weights = self._apply_regime_adjustments(optimal_weights)

            # Execute rebalancing
            rebalancing_actions = await self._execute_rebalancing(adjusted_weights)

            # Update portfolio state
            self.last_rebalance = datetime.now(timezone.utc)

            # Calculate portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics()

            return {
                "rebalance_timestamp": datetime.now(timezone.utc).isoformat(),
                "optimal_weights": optimal_weights,
                "adjusted_weights": adjusted_weights,
                "actions": rebalancing_actions,
                "portfolio_metrics": portfolio_metrics,
                "total_strategies": len(self.allocations),
                "allocated_capital": float(self.allocated_capital),
                "available_capital": float(self.available_capital),
            }

        except Exception as e:
            raise AllocationError(f"Portfolio rebalancing failed: {e}")

    async def _update_strategy_metrics(self) -> None:
        """Update performance metrics for all strategies."""
        for strategy_name, allocation in self.allocations.items():
            try:
                # Get strategy metrics
                strategy = allocation.strategy
                metrics = strategy.get_real_time_metrics()

                # Update allocation metrics
                allocation.trade_count = metrics.get("execution_count", 0)
                allocation.win_rate = metrics.get("win_rate", 0.0)

                # Calculate performance metrics if we have enough data
                if hasattr(strategy, "metrics") and strategy.metrics.total_trades > 10:
                    allocation.sharpe_ratio = float(strategy.metrics.sharpe_ratio or 0.0)
                    allocation.max_drawdown = float(strategy.metrics.max_drawdown or 0.0)

                    # Calculate Sortino ratio if available
                    if len(allocation.daily_returns) > 20:
                        negative_returns = [r for r in allocation.daily_returns if r < 0]
                        if negative_returns:
                            downside_deviation = np.std(negative_returns)
                            if downside_deviation > 0:
                                mean_return = np.mean(allocation.daily_returns)
                                allocation.sortino_ratio = mean_return / downside_deviation

            except Exception:
                # Log error but continue with other strategies
                continue

    async def _calculate_optimal_weights(self) -> dict[str, float]:
        """
        Calculate optimal portfolio weights using modern portfolio theory.

        Returns:
            Dictionary of optimal weights for each strategy
        """
        try:
            if len(self.allocations) == 0:
                return {}

            strategies = list(self.allocations.keys())
            n_strategies = len(strategies)

            # If only one strategy, allocate 100%
            if n_strategies == 1:
                return {strategies[0]: 1.0}

            # Prepare data for optimization
            returns_matrix = self._build_returns_matrix(strategies)

            if returns_matrix is None or len(returns_matrix) < 10:
                # Not enough data, use equal weights
                equal_weight = 1.0 / n_strategies
                return {name: equal_weight for name in strategies}

            # Calculate expected returns and covariance matrix
            expected_returns = np.mean(returns_matrix, axis=0)
            cov_matrix = np.cov(returns_matrix.T)

            # Optimize for maximum Sharpe ratio
            weights = self._optimize_sharpe_ratio(expected_returns, cov_matrix, n_strategies)

            # Apply constraints
            weights = self._apply_weight_constraints(weights)

            # Create result dictionary
            optimal_weights = {}
            for i, strategy_name in enumerate(strategies):
                optimal_weights[strategy_name] = float(weights[i])

            return optimal_weights

        except Exception:
            # Fallback to performance-based weights
            return self._calculate_performance_based_weights()

    def _build_returns_matrix(self, strategies: list[str]) -> np.ndarray | None:
        """Build returns matrix for portfolio optimization."""
        try:
            # Collect daily returns for each strategy
            returns_data = []
            min_length = float("inf")

            for strategy_name in strategies:
                allocation = self.allocations[strategy_name]
                if len(allocation.daily_returns) > 0:
                    returns_data.append(allocation.daily_returns)
                    min_length = min(min_length, len(allocation.daily_returns))
                else:
                    # Use default returns if no history
                    returns_data.append([0.001] * 30)  # 0.1% daily return
                    min_length = min(min_length, 30)

            if min_length == 0 or min_length == float("inf"):
                return None

            # Truncate all series to minimum length
            returns_matrix = np.array([returns[-min_length:] for returns in returns_data]).T

            return returns_matrix

        except Exception:
            return None

    def _optimize_sharpe_ratio(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray, n_strategies: int
    ) -> np.ndarray:
        """Optimize portfolio for maximum Sharpe ratio."""
        try:
            # Objective function (negative Sharpe ratio to minimize)
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

                if portfolio_volatility == 0:
                    return -portfolio_return

                sharpe_ratio = portfolio_return / portfolio_volatility
                return -sharpe_ratio  # Minimize negative Sharpe ratio

            # Constraints
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},  # Weights sum to 1
            ]

            # Bounds (min and max allocation per strategy)
            bounds = [
                (self.min_strategy_allocation, self.max_strategy_allocation)
                for _ in range(n_strategies)
            ]

            # Initial guess (equal weights)
            initial_weights = np.array([1.0 / n_strategies] * n_strategies)

            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if result.success:
                return result.x
            else:
                # Fallback to equal weights
                return initial_weights

        except Exception:
            # Fallback to equal weights
            return np.array([1.0 / n_strategies] * n_strategies)

    def _apply_weight_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply portfolio-level weight constraints."""
        # Ensure weights are positive and sum to 1
        weights = np.maximum(weights, self.min_strategy_allocation)
        weights = np.minimum(weights, self.max_strategy_allocation)

        # Normalize to sum to 1
        weights = weights / np.sum(weights)

        return weights

    def _calculate_performance_based_weights(self) -> dict[str, float]:
        """Calculate weights based on recent performance."""
        try:
            strategies = list(self.allocations.keys())

            if len(strategies) == 0:
                return {}

            # Calculate performance scores
            scores = {}
            for strategy_name in strategies:
                allocation = self.allocations[strategy_name]

                # Base score on Sharpe ratio with adjustments
                base_score = max(allocation.sharpe_ratio, 0.1)

                # Adjust for win rate
                win_rate_adjustment = 0.5 + (allocation.win_rate * 0.5)

                # Penalize high drawdown
                drawdown_penalty = max(0.1, 1.0 - allocation.max_drawdown)

                # Final score
                scores[strategy_name] = base_score * win_rate_adjustment * drawdown_penalty

            # Convert to weights
            total_score = sum(scores.values())
            if total_score == 0:
                # Equal weights fallback
                return {name: 1.0 / len(strategies) for name in strategies}

            return {name: score / total_score for name, score in scores.items()}

        except Exception:
            # Equal weights fallback
            strategies = list(self.allocations.keys())
            return {name: 1.0 / len(strategies) for name in strategies}

    def _apply_regime_adjustments(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply market regime-based weight adjustments."""
        try:
            regime_params = self.regime_allocations.get(self.current_regime, {})
            risk_multiplier = regime_params.get("risk_multiplier", 1.0)

            # Adjust weights based on regime
            adjusted_weights = {}
            for strategy_name, weight in weights.items():
                allocation = self.allocations[strategy_name]

                # Reduce allocation for high-risk strategies in bearish regimes
                if risk_multiplier < 1.0 and allocation.max_drawdown > 0.15:
                    adjusted_weight = weight * 0.8
                elif risk_multiplier > 1.0 and allocation.sharpe_ratio > 1.0:
                    adjusted_weight = weight * 1.2
                else:
                    adjusted_weight = weight

                adjusted_weights[strategy_name] = adjusted_weight

            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {
                    name: weight / total_weight for name, weight in adjusted_weights.items()
                }

            return adjusted_weights

        except Exception:
            return weights

    async def _execute_rebalancing(self, target_weights: dict[str, float]) -> list[dict[str, Any]]:
        """Execute the rebalancing by adjusting strategy allocations."""
        actions = []

        try:
            for strategy_name, target_weight in target_weights.items():
                allocation = self.allocations[strategy_name]
                current_weight = allocation.current_weight

                # Calculate weight difference
                weight_diff = target_weight - current_weight

                # Only rebalance if difference is significant
                if abs(weight_diff) > allocation.rebalance_threshold:
                    # Calculate capital adjustment
                    capital_change = self.total_capital * Decimal(str(weight_diff))

                    # Update allocation
                    old_capital = allocation.allocated_capital
                    allocation.allocated_capital += capital_change
                    allocation.current_weight = target_weight
                    allocation.target_weight = target_weight
                    allocation.last_rebalance = datetime.now(timezone.utc)

                    # Track action
                    actions.append(
                        {
                            "strategy": strategy_name,
                            "action": "increase" if weight_diff > 0 else "decrease",
                            "old_weight": current_weight,
                            "new_weight": target_weight,
                            "old_capital": float(old_capital),
                            "new_capital": float(allocation.allocated_capital),
                            "capital_change": float(capital_change),
                        }
                    )

            # Update portfolio totals
            self.allocated_capital = sum(
                allocation.allocated_capital for allocation in self.allocations.values()
            )
            self.available_capital = self.total_capital - self.allocated_capital

            return actions

        except Exception as e:
            raise AllocationError(f"Failed to execute rebalancing: {e}")

    async def _calculate_portfolio_metrics(self) -> dict[str, Any]:
        """Calculate comprehensive portfolio performance metrics."""
        try:
            if len(self.portfolio_returns) < 2:
                return self.portfolio_metrics

            returns = np.array(self.portfolio_returns)

            # Basic metrics
            total_return = np.prod(1 + returns) - 1
            mean_return = np.mean(returns)
            volatility = np.std(returns)

            # Annualized metrics (assuming daily returns)
            annualized_return = (1 + mean_return) ** 252 - 1
            annualized_volatility = volatility * np.sqrt(252)

            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            if annualized_volatility > 0:
                sharpe_ratio = (annualized_return - 0.02) / annualized_volatility
            else:
                sharpe_ratio = 0.0

            # Sortino ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_deviation = np.std(negative_returns) * np.sqrt(252)
                sortino_ratio = (
                    (annualized_return - 0.02) / downside_deviation
                    if downside_deviation > 0
                    else 0.0
                )
            else:
                sortino_ratio = sharpe_ratio

            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

            # Update portfolio metrics
            self.portfolio_metrics = {
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "volatility": float(annualized_volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "sortino_ratio": float(sortino_ratio),
                "max_drawdown": float(max_drawdown),
                "calmar_ratio": float(calmar_ratio),
            }

            return self.portfolio_metrics

        except Exception:
            return self.portfolio_metrics

    async def update_market_regime(self, new_regime: MarketRegime) -> None:
        """
        Update market regime and trigger rebalancing if needed.

        Args:
            new_regime: New market regime
        """
        if self.current_regime != new_regime:
            old_regime = self.current_regime
            self.current_regime = new_regime

            # Trigger rebalancing for significant regime changes
            regime_change_threshold = {
                (MarketRegime.BULL, MarketRegime.BEAR): True,
                (MarketRegime.BEAR, MarketRegime.BULL): True,
                (MarketRegime.NORMAL, MarketRegime.HIGH_VOLATILITY): True,
                (MarketRegime.HIGH_VOLATILITY, MarketRegime.NORMAL): True,
            }

            if regime_change_threshold.get((old_regime, new_regime), False):
                await self.rebalance_portfolio()

    async def remove_strategy(self, strategy_name: str, reason: str = "manual") -> bool:
        """
        Remove a strategy from the portfolio.

        Args:
            strategy_name: Name of strategy to remove
            reason: Reason for removal

        Returns:
            True if strategy was removed successfully
        """
        try:
            if strategy_name not in self.allocations:
                return False

            allocation = self.allocations[strategy_name]

            # Stop the strategy
            await allocation.strategy.stop()

            # Redistribute capital to remaining strategies
            freed_capital = allocation.allocated_capital
            self.available_capital += freed_capital

            # Remove allocation
            del self.allocations[strategy_name]

            # Trigger rebalancing to redistribute capital
            if len(self.allocations) > 0:
                await self.rebalance_portfolio()

            # Try to add a strategy from queue
            if self.strategy_queue and len(self.allocations) < self.max_strategies:
                next_strategy = self.strategy_queue.pop(0)
                await self.add_strategy(next_strategy)

            return True

        except Exception as e:
            raise AllocationError(f"Failed to remove strategy {strategy_name}: {e}")

    def get_allocation_status(self) -> dict[str, Any]:
        """
        Get current allocation status and portfolio summary.

        Returns:
            Comprehensive allocation status
        """
        try:
            strategy_summaries = {}
            for name, allocation in self.allocations.items():
                strategy_summaries[name] = {
                    "strategy_type": allocation.strategy.strategy_type.value,
                    "status": allocation.strategy.status.value,
                    "target_weight": allocation.target_weight,
                    "current_weight": allocation.current_weight,
                    "allocated_capital": float(allocation.allocated_capital),
                    "cumulative_pnl": float(allocation.cumulative_pnl),
                    "trade_count": allocation.trade_count,
                    "win_rate": allocation.win_rate,
                    "sharpe_ratio": allocation.sharpe_ratio,
                    "sortino_ratio": allocation.sortino_ratio,
                    "max_drawdown": allocation.max_drawdown,
                    "last_rebalance": allocation.last_rebalance.isoformat(),
                }

            return {
                "portfolio_summary": {
                    "total_capital": float(self.total_capital),
                    "allocated_capital": float(self.allocated_capital),
                    "available_capital": float(self.available_capital),
                    "portfolio_value": float(self.portfolio_value),
                    "number_of_strategies": len(self.allocations),
                    "strategies_in_queue": len(self.strategy_queue),
                    "current_regime": self.current_regime.value,
                    "last_rebalance": self.last_rebalance.isoformat(),
                },
                "portfolio_metrics": self.portfolio_metrics,
                "strategy_allocations": strategy_summaries,
                "risk_limits": {
                    "max_strategies": self.max_strategies,
                    "min_strategy_allocation": self.min_strategy_allocation,
                    "max_strategy_allocation": self.max_strategy_allocation,
                    "max_correlation_threshold": self.max_correlation_threshold,
                    "volatility_target": self.volatility_target,
                    "max_drawdown_limit": self.max_drawdown_limit,
                },
            }

        except Exception as e:
            raise AllocationError(f"Failed to get allocation status: {e}")

    async def should_rebalance(self) -> bool:
        """
        Determine if portfolio should be rebalanced.

        Returns:
            True if rebalancing is recommended
        """
        try:
            # Time-based rebalancing
            if datetime.now(timezone.utc) - self.last_rebalance > self.rebalance_frequency:
                return True

            # Weight drift rebalancing
            for allocation in self.allocations.values():
                weight_drift = abs(allocation.target_weight - allocation.current_weight)
                if weight_drift > allocation.rebalance_threshold:
                    return True

            # Performance-based rebalancing (any strategy with large drawdown)
            for allocation in self.allocations.values():
                if allocation.max_drawdown > 0.15:  # 15% drawdown threshold
                    return True

            # Portfolio risk-based rebalancing
            if self.portfolio_metrics.get("max_drawdown", 0) > self.max_drawdown_limit * 0.8:
                return True

            return False

        except Exception:
            return True  # Conservative approach - rebalance on error
