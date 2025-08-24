"""
Ensemble Strategy Implementation

This module implements an adaptive ensemble strategy that combines multiple trading
strategies with dynamic weight adjustment based on performance. It includes correlation
analysis, voting mechanisms, and strategy diversity maintenance.

Key Features:
- Multi-strategy ensemble with dynamic weights
- Performance-based weight adjustment
- Correlation analysis for strategy selection
- Real-time strategy performance tracking
- Multiple voting mechanisms (majority, weighted, confidence-based)
- Strategy diversity maintenance
- Risk-adjusted performance evaluation
"""

import asyncio
from decimal import Decimal
from typing import Any

import numpy as np
from scipy.stats import pearsonr

from src.core.exceptions import StrategyError

# Logger is provided by BaseStrategy (via BaseComponent)
# MANDATORY: Import from P-001
from src.core.types import MarketData, Position, Signal, SignalDirection, StrategyType

# MANDATORY: Import from P-010
from src.risk_management.regime_detection import MarketRegimeDetector

# MANDATORY: Import from P-011
from src.strategies.base import BaseStrategy

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution


class StrategyPerformanceTracker:
    """Tracks individual strategy performance within the ensemble."""

    def __init__(self, strategy_name: str):
        """Initialize performance tracker for a strategy."""
        self.strategy_name = strategy_name
        self.trades = []
        self.signals = []
        self.returns = []
        self.win_rate = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.total_return = 0.0
        self.volatility = 0.0
        self.correlation_with_market = 0.0

        # Rolling metrics
        self.rolling_window = 50
        self.recent_performance = []

    def add_signal(self, signal: Signal) -> None:
        """Add a signal from the strategy."""
        self.signals.append(
            {
                "timestamp": signal.timestamp,
                "direction": signal.direction,
                "confidence": signal.confidence,
                "symbol": signal.symbol,
            }
        )

    def add_trade_result(self, return_pct: float, trade_info: dict[str, Any]) -> None:
        """Add a completed trade result."""
        self.trades.append(
            {
                "return": return_pct,
                "timestamp": trade_info.get("timestamp"),
                "symbol": trade_info.get("symbol"),
                "side": trade_info.get("side"),
            }
        )

        self.returns.append(return_pct)
        self.recent_performance.append(return_pct)

        # Keep only recent performance for rolling metrics
        if len(self.recent_performance) > self.rolling_window:
            self.recent_performance = self.recent_performance[-self.rolling_window :]

        self._update_metrics()

    def _update_metrics(self) -> None:
        """Update performance metrics."""
        if not self.returns:
            return

        # Win rate
        winning_trades = sum(1 for r in self.returns if r > 0)
        self.win_rate = winning_trades / len(self.returns)

        # Total return
        self.total_return = sum(self.returns)

        # Volatility (annualized)
        if len(self.returns) > 1:
            self.volatility = np.std(self.returns) * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate)
        if self.volatility > 0:
            avg_return = np.mean(self.returns)
            self.sharpe_ratio = (avg_return * 252) / self.volatility
        else:
            self.sharpe_ratio = 0.0

        # Max drawdown
        cumulative_returns = np.cumsum(self.returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = peak - cumulative_returns
        self.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

    def get_recent_performance(self, window: int = 10) -> float:
        """Get average performance over recent trades."""
        if not self.recent_performance:
            return 0.0
        recent_trades = self.recent_performance[-window:]
        return np.mean(recent_trades)

    def get_performance_score(self) -> float:
        """Calculate a composite performance score."""
        if len(self.returns) < 5:
            return 0.5  # Neutral score for new strategies

        # Weighted combination of metrics
        win_rate_score = self.win_rate
        sharpe_score = max(0, min(1, (self.sharpe_ratio + 2) / 4))  # Normalize Sharpe to 0-1
        drawdown_score = max(0, 1 - self.max_drawdown)
        recent_score = max(0, min(1, (self.get_recent_performance() + 0.1) / 0.2))

        # Weighted average
        composite_score = (
            0.3 * win_rate_score + 0.3 * sharpe_score + 0.2 * drawdown_score + 0.2 * recent_score
        )

        return composite_score

    def get_metrics(self) -> dict[str, Any]:
        """Get all performance metrics."""
        return {
            "strategy_name": self.strategy_name,
            "total_trades": len(self.trades),
            "total_signals": len(self.signals),
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "recent_performance": self.get_recent_performance(),
            "performance_score": self.get_performance_score(),
            "correlation_with_market": self.correlation_with_market,
        }


class CorrelationAnalyzer:
    """Analyzes correlations between strategies for diversity maintenance."""

    def __init__(self, window_size: int = 50):
        """Initialize correlation analyzer."""
        self.window_size = window_size
        self.strategy_returns: dict[str, list[float]] = {}

    def add_strategy_return(self, strategy_name: str, return_value: float) -> None:
        """Add a return value for a strategy."""
        if strategy_name not in self.strategy_returns:
            self.strategy_returns[strategy_name] = []

        self.strategy_returns[strategy_name].append(return_value)

        # Keep only recent returns
        if len(self.strategy_returns[strategy_name]) > self.window_size:
            self.strategy_returns[strategy_name] = self.strategy_returns[strategy_name][
                -self.window_size :
            ]

    def calculate_correlation_matrix(self) -> dict[str, dict[str, float]]:
        """Calculate correlation matrix between all strategies."""
        strategies = list(self.strategy_returns.keys())
        correlation_matrix = {}

        for strategy1 in strategies:
            correlation_matrix[strategy1] = {}
            for strategy2 in strategies:
                if strategy1 == strategy2:
                    correlation_matrix[strategy1][strategy2] = 1.0
                else:
                    returns1 = self.strategy_returns[strategy1]
                    returns2 = self.strategy_returns[strategy2]

                    # Need sufficient data for correlation
                    min_length = min(len(returns1), len(returns2))
                    if min_length < 10:
                        correlation_matrix[strategy1][strategy2] = 0.0
                    else:
                        # Align returns to same length
                        aligned_returns1 = returns1[-min_length:]
                        aligned_returns2 = returns2[-min_length:]

                        try:
                            correlation, _ = pearsonr(aligned_returns1, aligned_returns2)
                            correlation_matrix[strategy1][strategy2] = (
                                correlation if not np.isnan(correlation) else 0.0
                            )
                        except:
                            correlation_matrix[strategy1][strategy2] = 0.0

        return correlation_matrix

    def get_diversity_score(self) -> float:
        """Calculate overall portfolio diversity score."""
        correlation_matrix = self.calculate_correlation_matrix()
        strategies = list(correlation_matrix.keys())

        if len(strategies) < 2:
            return 1.0  # Perfect diversity with single strategy

        # Calculate average correlation (excluding diagonal)
        total_correlation = 0.0
        count = 0

        for strategy1 in strategies:
            for strategy2 in strategies:
                if strategy1 != strategy2:
                    total_correlation += abs(correlation_matrix[strategy1][strategy2])
                    count += 1

        avg_correlation = total_correlation / count if count > 0 else 0.0

        # Convert to diversity score (lower correlation = higher diversity)
        diversity_score = 1.0 - avg_correlation
        return max(0.0, min(1.0, diversity_score))

    def identify_redundant_strategies(self, threshold: float = 0.8) -> list[tuple[str, str]]:
        """Identify pairs of highly correlated strategies."""
        correlation_matrix = self.calculate_correlation_matrix()
        redundant_pairs = []

        strategies = list(correlation_matrix.keys())
        for i, strategy1 in enumerate(strategies):
            for strategy2 in strategies[i + 1 :]:
                correlation = abs(correlation_matrix[strategy1][strategy2])
                if correlation > threshold:
                    redundant_pairs.append((strategy1, strategy2))

        return redundant_pairs


class VotingMechanism:
    """Implements different voting mechanisms for ensemble decisions."""

    @staticmethod
    def majority_vote(signals: list[tuple[str, Signal]]) -> Signal | None:
        """Simple majority voting mechanism."""
        if not signals:
            return None

        # Count votes by direction
        buy_votes = sum(1 for _, signal in signals if signal.direction == SignalDirection.BUY)
        sell_votes = sum(1 for _, signal in signals if signal.direction == SignalDirection.SELL)
        hold_votes = len(signals) - buy_votes - sell_votes

        # Determine majority direction
        if buy_votes > sell_votes and buy_votes > hold_votes:
            direction = SignalDirection.BUY
            confidence = buy_votes / len(signals)
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            direction = SignalDirection.SELL
            confidence = sell_votes / len(signals)
        else:
            return None  # No clear majority or majority hold

        # Use first signal as template
        template_signal = signals[0][1]

        return Signal(
            direction=direction,
            confidence=confidence,
            timestamp=template_signal.timestamp,
            symbol=template_signal.symbol,
            strategy_name="EnsembleMajority",
            metadata={
                "voting_method": "majority",
                "total_signals": len(signals),
                "buy_votes": buy_votes,
                "sell_votes": sell_votes,
                "contributing_strategies": [name for name, _ in signals],
            },
        )

    @staticmethod
    def weighted_vote(
        signals: list[tuple[str, Signal]], weights: dict[str, float]
    ) -> Signal | None:
        """Weighted voting based on strategy performance."""
        if not signals:
            return None

        total_buy_weight = 0.0
        total_sell_weight = 0.0
        total_weight = 0.0

        for strategy_name, signal in signals:
            weight = weights.get(strategy_name, 1.0)
            total_weight += weight

            if signal.direction == SignalDirection.BUY:
                total_buy_weight += weight * signal.confidence
            elif signal.direction == SignalDirection.SELL:
                total_sell_weight += weight * signal.confidence

        if total_weight == 0:
            return None

        # Normalize weights
        buy_strength = total_buy_weight / total_weight
        sell_strength = total_sell_weight / total_weight

        # Determine direction and confidence
        threshold = 0.3  # Minimum strength threshold
        if buy_strength > sell_strength and buy_strength > threshold:
            direction = SignalDirection.BUY
            confidence = buy_strength
        elif sell_strength > buy_strength and sell_strength > threshold:
            direction = SignalDirection.SELL
            confidence = sell_strength
        else:
            return None

        template_signal = signals[0][1]

        return Signal(
            direction=direction,
            confidence=confidence,
            timestamp=template_signal.timestamp,
            symbol=template_signal.symbol,
            strategy_name="EnsembleWeighted",
            metadata={
                "voting_method": "weighted",
                "total_signals": len(signals),
                "buy_strength": buy_strength,
                "sell_strength": sell_strength,
                "total_weight": total_weight,
                "strategy_weights": {name: weights.get(name, 1.0) for name, _ in signals},
                "contributing_strategies": [name for name, _ in signals],
            },
        )

    @staticmethod
    def confidence_weighted_vote(
        signals: list[tuple[str, Signal]], performance_scores: dict[str, float]
    ) -> Signal | None:
        """Voting weighted by both confidence and performance."""
        if not signals:
            return None

        total_buy_score = 0.0
        total_sell_score = 0.0
        total_score = 0.0

        for strategy_name, signal in signals:
            performance_score = performance_scores.get(strategy_name, 0.5)
            combined_weight = signal.confidence * performance_score
            total_score += combined_weight

            if signal.direction == SignalDirection.BUY:
                total_buy_score += combined_weight
            elif signal.direction == SignalDirection.SELL:
                total_sell_score += combined_weight

        if total_score == 0:
            return None

        # Normalize scores
        buy_strength = total_buy_score / total_score
        sell_strength = total_sell_score / total_score

        # Determine direction
        threshold = 0.4  # Higher threshold for confidence-weighted voting
        if buy_strength > sell_strength and buy_strength > threshold:
            direction = SignalDirection.BUY
            confidence = buy_strength
        elif sell_strength > buy_strength and sell_strength > threshold:
            direction = SignalDirection.SELL
            confidence = sell_strength
        else:
            return None

        template_signal = signals[0][1]

        return Signal(
            direction=direction,
            confidence=confidence,
            timestamp=template_signal.timestamp,
            symbol=template_signal.symbol,
            strategy_name="EnsembleConfidenceWeighted",
            metadata={
                "voting_method": "confidence_weighted",
                "total_signals": len(signals),
                "buy_strength": buy_strength,
                "sell_strength": sell_strength,
                "total_score": total_score,
                "strategy_scores": {
                    name: signal.confidence * performance_scores.get(name, 0.5)
                    for name, signal in signals
                },
                "contributing_strategies": [name for name, _ in signals],
            },
        )


class EnsembleStrategy(BaseStrategy):
    """
    Adaptive ensemble strategy that combines multiple trading strategies.

    This strategy dynamically weights component strategies based on performance,
    maintains diversity through correlation analysis, and uses sophisticated
    voting mechanisms to generate final signals.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the ensemble strategy."""
        # Ensure strategy type is set correctly
        config["strategy_type"] = StrategyType.HYBRID
        if "name" not in config:
            config["name"] = "Ensemble"

        super().__init__(config)

        # Ensemble configuration
        self.voting_method = config.get("voting_method", "confidence_weighted")
        self.max_strategies = config.get("max_strategies", 5)
        self.min_strategies = config.get("min_strategies", 2)
        self.rebalance_frequency = config.get("rebalance_frequency", 24)  # Hours
        self.correlation_threshold = config.get("correlation_threshold", 0.8)
        self.performance_window = config.get("performance_window", 50)
        self.diversity_weight = config.get("diversity_weight", 0.3)

        # Strategy management
        self.component_strategies: dict[str, BaseStrategy] = {}
        self.strategy_weights: dict[str, float] = {}
        self.performance_trackers: dict[str, StrategyPerformanceTracker] = {}
        self.correlation_analyzer = CorrelationAnalyzer(self.performance_window)
        self.regime_detector = MarketRegimeDetector(config.get("regime_detection", {}))

        # Ensemble state
        self.last_rebalance = None
        self.ensemble_signals: list[dict[str, Any]] = []
        self.ensemble_performance = StrategyPerformanceTracker("Ensemble")

        # Voting mechanism
        self.voting_mechanism = VotingMechanism()

        self.logger.info(
            "Ensemble strategy initialized",
            voting_method=self.voting_method,
            max_strategies=self.max_strategies,
            min_strategies=self.min_strategies,
        )

    def add_strategy(self, strategy: BaseStrategy, initial_weight: float = 1.0) -> None:
        """Add a component strategy to the ensemble."""
        try:
            strategy_name = strategy.name

            if len(self.component_strategies) >= self.max_strategies:
                self.logger.warning(
                    "Maximum strategies reached, cannot add strategy",
                    strategy=strategy_name,
                    max_strategies=self.max_strategies,
                )
                return

            self.component_strategies[strategy_name] = strategy
            self.strategy_weights[strategy_name] = initial_weight
            self.performance_trackers[strategy_name] = StrategyPerformanceTracker(strategy_name)

            self.logger.info(
                "Strategy added to ensemble",
                strategy=strategy_name,
                initial_weight=initial_weight,
                total_strategies=len(self.component_strategies),
            )

        except Exception as e:
            self.logger.error(
                "Error adding strategy to ensemble", strategy=strategy.name, error=str(e)
            )
            raise StrategyError(f"Failed to add strategy {strategy.name}: {e}")

    def remove_strategy(self, strategy_name: str) -> None:
        """Remove a strategy from the ensemble."""
        try:
            if strategy_name in self.component_strategies:
                del self.component_strategies[strategy_name]
                del self.strategy_weights[strategy_name]
                del self.performance_trackers[strategy_name]

                self.logger.info(
                    "Strategy removed from ensemble",
                    strategy=strategy_name,
                    remaining_strategies=len(self.component_strategies),
                )
            else:
                self.logger.warning("Strategy not found in ensemble", strategy=strategy_name)

        except Exception as e:
            self.logger.error(
                "Error removing strategy from ensemble", strategy=strategy_name, error=str(e)
            )

    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """Generate ensemble signals by combining component strategy signals."""
        try:
            if len(self.component_strategies) < self.min_strategies:
                self.logger.warning(
                    "Insufficient strategies for ensemble signal generation",
                    current_strategies=len(self.component_strategies),
                    min_required=self.min_strategies,
                )
                return []

            # Get current market regime
            current_regime = await self.regime_detector.detect_comprehensive_regime([data])

            # Collect signals from all component strategies
            component_signals = []

            for strategy_name, strategy in self.component_strategies.items():
                try:
                    signals = await strategy.generate_signals(data)
                    for signal in signals:
                        component_signals.append((strategy_name, signal))
                        # Track signal for performance analysis
                        self.performance_trackers[strategy_name].add_signal(signal)
                except Exception as e:
                    self.logger.error(
                        "Error getting signals from component strategy",
                        strategy=strategy_name,
                        error=str(e),
                    )

            if not component_signals:
                self.logger.debug("No signals from component strategies")
                return []

            # Check if rebalancing is needed
            await self._check_and_rebalance()

            # Generate ensemble signal using voting mechanism
            ensemble_signal = await self._vote_on_signals(component_signals, current_regime)

            if ensemble_signal:
                # Store ensemble signal for performance tracking
                signal_record = {
                    "timestamp": data.timestamp,
                    "symbol": data.symbol,
                    "ensemble_signal": ensemble_signal.model_dump(),
                    "component_signals": [
                        (name, signal.model_dump()) for name, signal in component_signals
                    ],
                    "regime": current_regime.value,
                    "strategy_weights": self.strategy_weights.copy(),
                    "voting_method": self.voting_method,
                }
                self.ensemble_signals.append(signal_record)

                # Keep only recent signals
                if len(self.ensemble_signals) > 1000:
                    self.ensemble_signals = self.ensemble_signals[-1000:]

                return [ensemble_signal]

            return []

        except Exception as e:
            self.logger.error("Error generating ensemble signals", symbol=data.symbol, error=str(e))
            return []

    async def _vote_on_signals(
        self, component_signals: list[tuple[str, Signal]], regime: Any
    ) -> Signal | None:
        """Use voting mechanism to generate ensemble signal."""
        try:
            if not component_signals:
                return None

            # Apply regime-based filtering
            filtered_signals = await self._filter_signals_by_regime(component_signals, regime)

            if not filtered_signals:
                return None

            # Apply voting mechanism
            if self.voting_method == "majority":
                return self.voting_mechanism.majority_vote(filtered_signals)
            elif self.voting_method == "weighted":
                return self.voting_mechanism.weighted_vote(filtered_signals, self.strategy_weights)
            elif self.voting_method == "confidence_weighted":
                performance_scores = {
                    name: tracker.get_performance_score()
                    for name, tracker in self.performance_trackers.items()
                }
                return self.voting_mechanism.confidence_weighted_vote(
                    filtered_signals, performance_scores
                )
            else:
                self.logger.warning(
                    "Unknown voting method, using confidence_weighted", method=self.voting_method
                )
                performance_scores = {
                    name: tracker.get_performance_score()
                    for name, tracker in self.performance_trackers.items()
                }
                return self.voting_mechanism.confidence_weighted_vote(
                    filtered_signals, performance_scores
                )

        except Exception as e:
            self.logger.error("Error in signal voting", error=str(e))
            return None

    async def _filter_signals_by_regime(
        self, signals: list[tuple[str, Signal]], regime: Any
    ) -> list[tuple[str, Signal]]:
        """Filter signals based on market regime and strategy suitability."""
        # For now, return all signals - in production, you might want to filter
        # based on which strategies perform better in specific regimes
        return signals

    async def _check_and_rebalance(self) -> None:
        """Check if rebalancing is needed and perform it."""
        from datetime import datetime, timedelta

        try:
            current_time = datetime.now()

            if self.last_rebalance is None or current_time - self.last_rebalance > timedelta(
                hours=self.rebalance_frequency
            ):
                await self._rebalance_strategies()
                self.last_rebalance = current_time

        except Exception as e:
            self.logger.error("Error in rebalancing check", error=str(e))

    async def _rebalance_strategies(self) -> None:
        """Rebalance strategy weights based on performance and diversity."""
        try:
            if len(self.component_strategies) < 2:
                return

            self.logger.info("Starting strategy rebalancing")

            # Calculate performance-based weights
            performance_weights = {}
            total_performance = 0.0

            for strategy_name, tracker in self.performance_trackers.items():
                score = tracker.get_performance_score()
                performance_weights[strategy_name] = score
                total_performance += score

            # Normalize performance weights
            if total_performance > 0:
                for strategy_name in performance_weights:
                    performance_weights[strategy_name] /= total_performance
            else:
                # Equal weights if no performance data
                equal_weight = 1.0 / len(performance_weights)
                for strategy_name in performance_weights:
                    performance_weights[strategy_name] = equal_weight

            # Apply diversity adjustment
            diversity_score = self.correlation_analyzer.get_diversity_score()

            # Identify redundant strategies
            redundant_pairs = self.correlation_analyzer.identify_redundant_strategies(
                self.correlation_threshold
            )

            # Reduce weights for redundant strategies
            for strategy1, strategy2 in redundant_pairs:
                # Reduce weight of worse performing strategy
                if strategy1 in performance_weights and strategy2 in performance_weights:
                    if performance_weights[strategy1] < performance_weights[strategy2]:
                        performance_weights[strategy1] *= 0.7
                    else:
                        performance_weights[strategy2] *= 0.7

            # Final weight calculation
            final_weights = {}
            total_weight = sum(performance_weights.values())

            if total_weight > 0:
                for strategy_name, perf_weight in performance_weights.items():
                    final_weights[strategy_name] = perf_weight / total_weight
            else:
                equal_weight = 1.0 / len(performance_weights)
                for strategy_name in performance_weights:
                    final_weights[strategy_name] = equal_weight

            # Update strategy weights
            self.strategy_weights = final_weights

            self.logger.info(
                "Strategy rebalancing completed",
                new_weights=self.strategy_weights,
                diversity_score=diversity_score,
                redundant_pairs=len(redundant_pairs),
            )

        except Exception as e:
            self.logger.error("Error during strategy rebalancing", error=str(e))

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate ensemble signal."""
        try:
            # Basic signal validation
            if signal.confidence < self.config.min_confidence:
                return False

            if signal.direction == SignalDirection.HOLD:
                return False

            # Check metadata
            if not signal.metadata or "voting_method" not in signal.metadata:
                return False

            # Ensure sufficient strategy participation
            contributing_strategies = signal.metadata.get("contributing_strategies", [])
            if len(contributing_strategies) < self.min_strategies:
                self.logger.warning(
                    "Insufficient strategy participation",
                    contributing=len(contributing_strategies),
                    required=self.min_strategies,
                )
                return False

            return True

        except Exception as e:
            self.logger.error("Error validating ensemble signal", error=str(e))
            return False

    def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size based on ensemble confidence and diversity."""
        try:
            base_size = Decimal(str(self.config.position_size_pct))

            # Adjust based on signal confidence
            confidence_multiplier = Decimal(str(signal.confidence))

            # Adjust based on strategy participation
            contributing_strategies = signal.metadata.get("contributing_strategies", [])
            participation_ratio = len(contributing_strategies) / len(self.component_strategies)
            participation_multiplier = Decimal(str(0.5 + 0.5 * participation_ratio))

            # Adjust based on diversity score
            diversity_score = self.correlation_analyzer.get_diversity_score()
            diversity_multiplier = Decimal(str(0.8 + 0.4 * diversity_score))

            final_size = (
                base_size * confidence_multiplier * participation_multiplier * diversity_multiplier
            )

            # Ensure reasonable bounds
            max_size = Decimal("0.08")  # 8% maximum
            min_size = Decimal("0.005")  # 0.5% minimum

            return max(min_size, min(max_size, final_size))

        except Exception as e:
            self.logger.error("Error calculating ensemble position size", error=str(e))
            return Decimal("0.01")

    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Determine if position should be closed using ensemble logic."""
        try:
            # Basic stop-loss and take-profit
            current_price = data.current_price
            entry_price = position.entry_price

            if position.side.value == "buy":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # Stop loss
            if pnl_pct <= -self.config.stop_loss_pct:
                return True

            # Take profit
            if pnl_pct >= self.config.take_profit_pct:
                return True

            # Ensemble-based exit logic
            # Check if majority of strategies are signaling opposite direction
            opposing_signals = 0
            total_signals = 0

            for _strategy_name, strategy in self.component_strategies.items():
                try:
                    asyncio.create_task(strategy.generate_signals(data))
                    # Note: This is simplified - in practice you'd want to cache recent signals
                    # rather than generate new ones for exit decisions
                    total_signals += 1
                    # Check for opposing signals (simplified logic)
                    if hasattr(strategy, "last_signal_direction"):
                        if (
                            position.side.value == "buy"
                            and strategy.last_signal_direction == SignalDirection.SELL
                        ):
                            opposing_signals += 1
                        elif (
                            position.side.value == "sell"
                            and strategy.last_signal_direction == SignalDirection.BUY
                        ):
                            opposing_signals += 1
                except:
                    continue

            # Exit if majority of strategies are signaling opposite direction
            if total_signals > 0 and opposing_signals / total_signals > 0.6:
                self.logger.info(
                    "Ensemble exit signal triggered",
                    symbol=position.symbol,
                    opposing_ratio=opposing_signals / total_signals,
                )
                return True

            return False

        except Exception as e:
            self.logger.error(
                "Error in ensemble exit decision", symbol=position.symbol, error=str(e)
            )
            return False

    async def _on_start(self) -> None:
        """Start all component strategies."""
        self.logger.info("Starting ensemble strategy components")

        for strategy_name, strategy in self.component_strategies.items():
            try:
                await strategy.start()
                self.logger.info("Component strategy started", strategy=strategy_name)
            except Exception as e:
                self.logger.error(
                    "Failed to start component strategy", strategy=strategy_name, error=str(e)
                )

    async def _on_stop(self) -> None:
        """Stop all component strategies."""
        self.logger.info("Stopping ensemble strategy components")

        for strategy_name, strategy in self.component_strategies.items():
            try:
                await strategy.stop()
                self.logger.info("Component strategy stopped", strategy=strategy_name)
            except Exception as e:
                self.logger.error(
                    "Failed to stop component strategy", strategy=strategy_name, error=str(e)
                )

    def update_strategy_performance(
        self, strategy_name: str, return_pct: float, trade_info: dict[str, Any]
    ) -> None:
        """Update performance tracking for a component strategy."""
        try:
            if strategy_name in self.performance_trackers:
                self.performance_trackers[strategy_name].add_trade_result(return_pct, trade_info)
                self.correlation_analyzer.add_strategy_return(strategy_name, return_pct)

                self.logger.debug(
                    "Strategy performance updated",
                    strategy=strategy_name,
                    return_pct=return_pct,
                    new_score=self.performance_trackers[strategy_name].get_performance_score(),
                )
        except Exception as e:
            self.logger.error(
                "Error updating strategy performance", strategy=strategy_name, error=str(e)
            )

    def get_ensemble_statistics(self) -> dict[str, Any]:
        """Get comprehensive ensemble statistics."""
        try:
            strategy_metrics = {}
            for name, tracker in self.performance_trackers.items():
                strategy_metrics[name] = tracker.get_metrics()

            correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix()
            diversity_score = self.correlation_analyzer.get_diversity_score()
            redundant_pairs = self.correlation_analyzer.identify_redundant_strategies()

            return {
                "ensemble_config": {
                    "voting_method": self.voting_method,
                    "max_strategies": self.max_strategies,
                    "min_strategies": self.min_strategies,
                    "correlation_threshold": self.correlation_threshold,
                },
                "component_strategies": list(self.component_strategies.keys()),
                "strategy_weights": self.strategy_weights,
                "strategy_metrics": strategy_metrics,
                "correlation_matrix": correlation_matrix,
                "diversity_score": diversity_score,
                "redundant_pairs": redundant_pairs,
                "ensemble_performance": self.ensemble_performance.get_metrics(),
                "total_ensemble_signals": len(self.ensemble_signals),
                "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
            }

        except Exception as e:
            self.logger.error("Error getting ensemble statistics", error=str(e))
            return {"error": str(e)}
