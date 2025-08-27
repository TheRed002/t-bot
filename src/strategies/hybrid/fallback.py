"""
Intelligent Fallback Strategy Implementation

This module implements intelligent fallback mechanisms for robust trading when
primary strategies fail or perform poorly. It provides graceful degradation to
simpler strategies, market condition-based triggers, and recovery mechanisms.

Key Features:
- Failure detection for primary strategies
- Graceful degradation to simpler strategies
- Market condition-based fallback triggers
- Recovery mechanisms when primary strategies improve
- Emergency safe mode trading
- Performance monitoring of fallback activations
- Health monitoring and automated switching
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np

from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import MarketData, Position, Signal, SignalDirection, StrategyType

# MANDATORY: Import from P-010
from src.risk_management.regime_detection import MarketRegimeDetector

# MANDATORY: Import from P-011
from src.strategies.base import BaseStrategy

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution


class FallbackMode(Enum):
    """Fallback operation modes."""

    PRIMARY = "primary"  # Normal operation with primary strategy
    DEGRADED = "degraded"  # Degraded mode with simpler strategy
    SAFE_MODE = "safe_mode"  # Emergency safe mode with minimal trading
    RECOVERY = "recovery"  # Recovery testing mode
    SHUTDOWN = "shutdown"  # Complete shutdown mode


class FailureType(Enum):
    """Types of failures that can trigger fallback."""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    TECHNICAL_ERROR = "technical_error"
    HIGH_DRAWDOWN = "high_drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    LOW_CONFIDENCE = "low_confidence"
    MARKET_VOLATILITY = "market_volatility"
    DATA_QUALITY = "data_quality"
    STRATEGY_TIMEOUT = "strategy_timeout"


class FailureDetector:
    """Detects various types of strategy failures."""

    def __init__(self, config: dict[str, Any]):
        """Initialize failure detector."""
        self.config = config

        # Performance thresholds
        self.max_drawdown_threshold = config.get("max_drawdown_threshold", 0.15)  # 15%
        self.min_win_rate_threshold = config.get("min_win_rate_threshold", 0.3)  # 30%
        self.consecutive_loss_threshold = config.get("consecutive_loss_threshold", 5)
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.4)
        self.performance_window = config.get("performance_window", 20)

        # Technical thresholds
        self.max_error_rate = config.get("max_error_rate", 0.2)  # 20% error rate
        self.timeout_threshold = config.get("timeout_threshold", 30)  # 30 seconds

        # Market condition thresholds
        self.max_volatility_threshold = config.get(
            "max_volatility_threshold", 0.5
        )  # 50% annualized

        # State tracking
        self.recent_trades = []
        self.recent_errors = []
        self.recent_timeouts = []
        self.last_signals = []

    def add_trade_result(self, return_pct: float, timestamp: datetime) -> None:
        """Add a trade result for analysis."""
        self.recent_trades.append(
            {"return": return_pct, "timestamp": timestamp, "is_win": return_pct > 0}
        )

        # Keep only recent trades
        if len(self.recent_trades) > self.performance_window * 2:
            self.recent_trades = self.recent_trades[-self.performance_window :]

    def add_error(self, error_type: str, timestamp: datetime) -> None:
        """Add an error occurrence for analysis."""
        self.recent_errors.append({"error_type": error_type, "timestamp": timestamp})

        # Keep only recent errors (last 24 hours)
        cutoff_time = timestamp - timedelta(hours=24)
        self.recent_errors = [
            error for error in self.recent_errors if error["timestamp"] > cutoff_time
        ]

    def add_timeout(self, duration: float, timestamp: datetime) -> None:
        """Add a timeout occurrence for analysis."""
        self.recent_timeouts.append({"duration": duration, "timestamp": timestamp})

        # Keep only recent timeouts (last hour)
        cutoff_time = timestamp - timedelta(hours=1)
        self.recent_timeouts = [
            timeout for timeout in self.recent_timeouts if timeout["timestamp"] > cutoff_time
        ]

    def add_signal(self, signal: Signal) -> None:
        """Add a signal for confidence analysis."""
        self.last_signals.append(signal)

        # Keep only recent signals
        if len(self.last_signals) > 50:
            self.last_signals = self.last_signals[-50:]

    def detect_performance_degradation(self) -> dict[str, Any]:
        """Detect performance degradation issues."""
        if len(self.recent_trades) < self.performance_window:
            return {"detected": False, "reason": "insufficient_data"}

        recent_window = self.recent_trades[-self.performance_window :]

        # Calculate metrics
        returns = [trade["return"] for trade in recent_window]
        wins = [trade for trade in recent_window if trade["is_win"]]

        # Win rate check
        win_rate = len(wins) / len(recent_window)
        if win_rate < self.min_win_rate_threshold:
            return {
                "detected": True,
                "failure_type": FailureType.PERFORMANCE_DEGRADATION,
                "reason": "low_win_rate",
                "win_rate": win_rate,
                "threshold": self.min_win_rate_threshold,
            }

        # Drawdown check
        cumulative_returns = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = peak - cumulative_returns
        max_drawdown = np.max(drawdown)

        if max_drawdown > self.max_drawdown_threshold:
            return {
                "detected": True,
                "failure_type": FailureType.HIGH_DRAWDOWN,
                "reason": "high_drawdown",
                "max_drawdown": max_drawdown,
                "threshold": self.max_drawdown_threshold,
            }

        # Consecutive losses check
        consecutive_losses = 0
        for trade in reversed(recent_window):
            if not trade["is_win"]:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= self.consecutive_loss_threshold:
            return {
                "detected": True,
                "failure_type": FailureType.CONSECUTIVE_LOSSES,
                "reason": "consecutive_losses",
                "consecutive_losses": consecutive_losses,
                "threshold": self.consecutive_loss_threshold,
            }

        return {"detected": False}

    def detect_technical_issues(self) -> dict[str, Any]:
        """Detect technical issues that could cause failures."""
        current_time = datetime.now(timezone.utc)

        # Error rate check
        recent_errors_1h = [
            error
            for error in self.recent_errors
            if current_time - error["timestamp"] < timedelta(hours=1)
        ]

        if len(recent_errors_1h) > 0:
            # Calculate error rate (errors per hour)
            error_rate = len(recent_errors_1h)  # Already filtered to 1 hour
            if error_rate > self.max_error_rate * 60:  # Convert to errors per hour
                return {
                    "detected": True,
                    "failure_type": FailureType.TECHNICAL_ERROR,
                    "reason": "high_error_rate",
                    "error_rate": error_rate,
                    "threshold": self.max_error_rate * 60,
                }

        # Timeout check
        recent_timeouts_1h = [
            timeout
            for timeout in self.recent_timeouts
            if current_time - timeout["timestamp"] < timedelta(hours=1)
        ]

        if len(recent_timeouts_1h) > 5:  # More than 5 timeouts in an hour
            avg_timeout = np.mean([t["duration"] for t in recent_timeouts_1h])
            if avg_timeout > self.timeout_threshold:
                return {
                    "detected": True,
                    "failure_type": FailureType.STRATEGY_TIMEOUT,
                    "reason": "frequent_timeouts",
                    "timeout_count": len(recent_timeouts_1h),
                    "avg_timeout": avg_timeout,
                    "threshold": self.timeout_threshold,
                }

        return {"detected": False}

    def detect_confidence_issues(self) -> dict[str, Any]:
        """Detect low confidence signals."""
        if len(self.last_signals) < 10:
            return {"detected": False, "reason": "insufficient_signals"}

        recent_signals = self.last_signals[-10:]
        avg_confidence = np.mean([signal.confidence for signal in recent_signals])

        if avg_confidence < self.min_confidence_threshold:
            return {
                "detected": True,
                "failure_type": FailureType.LOW_CONFIDENCE,
                "reason": "low_average_confidence",
                "avg_confidence": avg_confidence,
                "threshold": self.min_confidence_threshold,
            }

        return {"detected": False}

    def detect_market_conditions(self, market_data: list[MarketData]) -> dict[str, Any]:
        """Detect adverse market conditions."""
        if len(market_data) < 20:
            return {"detected": False, "reason": "insufficient_market_data"}

        # Calculate market volatility
        prices = [float(data.price) for data in market_data[-20:]]
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        if volatility > self.max_volatility_threshold:
            return {
                "detected": True,
                "failure_type": FailureType.MARKET_VOLATILITY,
                "reason": "high_market_volatility",
                "volatility": volatility,
                "threshold": self.max_volatility_threshold,
            }

        return {"detected": False}


class SafeModeStrategy:
    """Emergency safe mode strategy with minimal risk."""

    def __init__(self, config: dict[str, Any]):
        """Initialize safe mode strategy."""
        self.config = config
        self.position_limit = config.get("safe_mode_position_limit", 0.01)  # 1% max position
        self.confidence_threshold = config.get("safe_mode_confidence_threshold", 0.8)
        self.max_positions = config.get("safe_mode_max_positions", 1)
        self.logger = get_logger(__name__)

    async def generate_signal(self, data: MarketData, price_history: list[float]) -> Signal | None:
        """Generate conservative signals in safe mode."""
        try:
            if len(price_history) < 20:
                return None

            # Very conservative signal generation
            # Only trade on strong trends with high confidence

            # Calculate simple moving averages
            ma_short = np.mean(price_history[-5:])
            ma_long = np.mean(price_history[-20:])

            # Only trade on clear trend
            trend_strength = abs(ma_short - ma_long) / ma_long

            if trend_strength > 0.02:  # 2% trend required
                if ma_short > ma_long * 1.02:  # Strong uptrend
                    confidence = min(trend_strength * 10, 1.0)
                    if confidence > self.confidence_threshold:
                        return Signal(
                            direction=SignalDirection.BUY,
                            confidence=confidence,
                            timestamp=data.timestamp,
                            symbol=data.symbol,
                            strategy_name="SafeMode",
                            metadata={
                                "mode": "safe_mode",
                                "trend_strength": trend_strength,
                                "ma_short": ma_short,
                                "ma_long": ma_long,
                            },
                        )
                elif ma_short < ma_long * 0.98:  # Strong downtrend
                    confidence = min(trend_strength * 10, 1.0)
                    if confidence > self.confidence_threshold:
                        return Signal(
                            direction=SignalDirection.SELL,
                            confidence=confidence,
                            timestamp=data.timestamp,
                            symbol=data.symbol,
                            strategy_name="SafeMode",
                            metadata={
                                "mode": "safe_mode",
                                "trend_strength": trend_strength,
                                "ma_short": ma_short,
                                "ma_long": ma_long,
                            },
                        )

            return None

        except Exception as e:
            self.logger.error("Error in safe mode signal generation", error=str(e))
            return None


class DegradedModeStrategy:
    """Degraded mode strategy with simplified logic."""

    def __init__(self, config: dict[str, Any]):
        """Initialize degraded mode strategy."""
        self.config = config
        self.position_limit = config.get("degraded_mode_position_limit", 0.02)  # 2% max position
        self.confidence_threshold = config.get("degraded_mode_confidence_threshold", 0.6)
        self.logger = get_logger(__name__)

    async def generate_signal(self, data: MarketData, price_history: list[float]) -> Signal | None:
        """Generate simplified signals in degraded mode."""
        try:
            if len(price_history) < 15:
                return None

            # Simplified technical analysis
            float(data.price)

            # Moving averages
            ma_5 = np.mean(price_history[-5:])
            ma_15 = np.mean(price_history[-15:])

            # RSI calculation (simplified)
            if len(price_history) >= 14:
                deltas = np.diff(price_history[-14:])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)

                if avg_loss > 0:
                    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
                else:
                    rsi = 100
            else:
                rsi = 50  # Neutral

            # Simple signal logic
            signal_strength = 0.0
            direction = SignalDirection.HOLD

            # Moving average crossover
            if ma_5 > ma_15 * 1.01:  # 1% threshold
                signal_strength += 0.3
                direction = SignalDirection.BUY
            elif ma_5 < ma_15 * 0.99:
                signal_strength += 0.3
                direction = SignalDirection.SELL

            # RSI confirmation
            if direction == SignalDirection.BUY and rsi < 70:
                signal_strength += 0.2
            elif direction == SignalDirection.SELL and rsi > 30:
                signal_strength += 0.2

            # Volume confirmation (if available)
            if data.volume and len(price_history) >= 5:
                recent_volumes = [float(data.volume)] * 5  # Simplified - would need volume history
                avg_volume = np.mean(recent_volumes)
                if float(data.volume) > avg_volume * 1.2:
                    signal_strength += 0.1

            if signal_strength >= self.confidence_threshold and direction != SignalDirection.HOLD:
                return Signal(
                    direction=direction,
                    confidence=signal_strength,
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    strategy_name="DegradedMode",
                    metadata={
                        "mode": "degraded_mode",
                        "ma_5": ma_5,
                        "ma_15": ma_15,
                        "rsi": rsi,
                        "signal_strength": signal_strength,
                    },
                )

            return None

        except Exception as e:
            self.logger.error("Error in degraded mode signal generation", error=str(e))
            return None


class FallbackStrategy(BaseStrategy):
    """
    Intelligent fallback strategy with automatic failure detection and recovery.

    This strategy monitors primary strategy health and automatically switches to
    fallback modes when issues are detected. It provides graceful degradation
    and recovery mechanisms for robust trading operations.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the fallback strategy."""
        # Ensure strategy type is set correctly
        config["strategy_type"] = StrategyType.HYBRID
        if "name" not in config:
            config["name"] = "Fallback"

        super().__init__(config)

        # Primary strategy
        self.primary_strategy: BaseStrategy | None = None

        # Fallback components
        self.failure_detector = FailureDetector(config.get("failure_detection", {}))
        self.safe_mode_strategy = SafeModeStrategy(config.get("safe_mode", {}))
        self.degraded_mode_strategy = DegradedModeStrategy(config.get("degraded_mode", {}))
        self.regime_detector = MarketRegimeDetector(config.get("regime_detection", {}))

        # Fallback state
        self.current_mode = FallbackMode.PRIMARY
        self.mode_history = []
        self.last_mode_change = datetime.now(timezone.utc)
        self.failure_count = 0
        self.recovery_attempts = 0

        # Configuration
        self.recovery_test_duration = timedelta(
            minutes=config.get("recovery_test_duration_minutes", 30)
        )
        self.max_failure_count = config.get("max_failure_count", 3)
        self.recovery_performance_threshold = config.get("recovery_performance_threshold", 0.6)

        # Data storage
        self.price_history: dict[str, list[float]] = {}
        self.market_data_history: list[MarketData] = []

        self.logger.info(
            "Fallback strategy initialized",
            max_failure_count=self.max_failure_count,
            recovery_test_duration=self.recovery_test_duration.total_seconds(),
        )

    def set_primary_strategy(self, strategy: BaseStrategy) -> None:
        """Set the primary strategy to monitor and fallback from."""
        self.primary_strategy = strategy
        self.logger.info("Primary strategy set", strategy=strategy.name)

    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """Generate signals with fallback logic."""
        try:
            symbol = data.symbol

            # Update data history
            if symbol not in self.price_history:
                self.price_history[symbol] = []

            self.price_history[symbol].append(float(data.price))
            self.market_data_history.append(data)

            # Keep only recent history
            self.price_history[symbol] = self.price_history[symbol][-200:]
            self.market_data_history = self.market_data_history[-200:]

            # Check for failures and update mode
            await self._check_and_update_mode(data)

            # Generate signal based on current mode
            signal = await self._generate_signal_for_current_mode(data)

            if signal:
                # Track signal for failure detection
                self.failure_detector.add_signal(signal)

                # Record mode information in signal metadata
                if not signal.metadata:
                    signal.metadata = {}
                signal.metadata.update(
                    {
                        "fallback_mode": self.current_mode.value,
                        "failure_count": self.failure_count,
                        "recovery_attempts": self.recovery_attempts,
                    }
                )

                return [signal]

            return []

        except Exception as e:
            self.logger.error(
                "Error in fallback signal generation", symbol=data.symbol, error=str(e)
            )
            # Record error for failure detection
            self.failure_detector.add_error("signal_generation", datetime.now(timezone.utc))
            return []

    async def _check_and_update_mode(self, data: MarketData) -> None:
        """Check for failures and update fallback mode."""
        try:
            current_time = datetime.now(timezone.utc)

            # Skip if recently changed mode (stability period)
            if current_time - self.last_mode_change < timedelta(minutes=5):
                return

            # Check different types of failures
            failure_detected = False
            failure_info = None

            # Performance degradation check
            perf_failure = self.failure_detector.detect_performance_degradation()
            if perf_failure["detected"]:
                failure_detected = True
                failure_info = perf_failure

            # Technical issues check
            if not failure_detected:
                tech_failure = self.failure_detector.detect_technical_issues()
                if tech_failure["detected"]:
                    failure_detected = True
                    failure_info = tech_failure

            # Confidence issues check
            if not failure_detected:
                conf_failure = self.failure_detector.detect_confidence_issues()
                if conf_failure["detected"]:
                    failure_detected = True
                    failure_info = conf_failure

            # Market condition check
            if not failure_detected:
                market_failure = self.failure_detector.detect_market_conditions(
                    self.market_data_history
                )
                if market_failure["detected"]:
                    failure_detected = True
                    failure_info = market_failure

            # Handle mode transitions
            if failure_detected:
                await self._handle_failure(failure_info)
            elif self.current_mode in [FallbackMode.DEGRADED, FallbackMode.SAFE_MODE]:
                await self._check_recovery_conditions()

        except Exception as e:
            self.logger.error("Error checking fallback conditions", error=str(e))

    async def _handle_failure(self, failure_info: dict[str, Any]) -> None:
        """Handle detected failure and switch modes."""
        try:
            self.failure_count += 1
            failure_type = failure_info.get("failure_type")

            self.logger.warning(
                "Failure detected, initiating fallback",
                failure_type=failure_type.value if failure_type else "unknown",
                failure_info=failure_info,
                failure_count=self.failure_count,
                current_mode=self.current_mode.value,
            )

            # Determine new mode based on failure severity and count
            new_mode = self._determine_fallback_mode(failure_info)

            if new_mode != self.current_mode:
                await self._switch_mode(new_mode, failure_info)

        except Exception as e:
            self.logger.error("Error handling failure", error=str(e))

    def _determine_fallback_mode(self, failure_info: dict[str, Any]) -> FallbackMode:
        """Determine appropriate fallback mode based on failure type."""
        failure_type = failure_info.get("failure_type")

        # Critical failures go straight to safe mode
        critical_failures = [
            FailureType.TECHNICAL_ERROR,
            FailureType.HIGH_DRAWDOWN,
            FailureType.STRATEGY_TIMEOUT,
        ]

        if failure_type in critical_failures or self.failure_count >= self.max_failure_count:
            return FallbackMode.SAFE_MODE

        # Market volatility might require safe mode
        if failure_type == FailureType.MARKET_VOLATILITY:
            volatility = failure_info.get("volatility", 0)
            if volatility > 0.7:  # Very high volatility
                return FallbackMode.SAFE_MODE
            else:
                return FallbackMode.DEGRADED

        # Performance issues typically go to degraded mode first
        if failure_type in [
            FailureType.PERFORMANCE_DEGRADATION,
            FailureType.CONSECUTIVE_LOSSES,
            FailureType.LOW_CONFIDENCE,
        ]:
            if self.current_mode == FallbackMode.PRIMARY:
                return FallbackMode.DEGRADED
            else:
                return FallbackMode.SAFE_MODE

        # Default to degraded mode
        return FallbackMode.DEGRADED

    async def _switch_mode(self, new_mode: FallbackMode, failure_info: dict[str, Any]) -> None:
        """Switch to new fallback mode."""
        try:
            old_mode = self.current_mode
            self.current_mode = new_mode
            self.last_mode_change = datetime.now(timezone.utc)

            # Record mode change
            mode_change = {
                "timestamp": self.last_mode_change,
                "from_mode": old_mode.value,
                "to_mode": new_mode.value,
                "failure_info": failure_info,
                "failure_count": self.failure_count,
            }
            self.mode_history.append(mode_change)

            # Keep only recent mode history
            if len(self.mode_history) > 100:
                self.mode_history = self.mode_history[-100:]

            self.logger.warning(
                "Fallback mode switched",
                from_mode=old_mode.value,
                to_mode=new_mode.value,
                failure_type=(
                    failure_info.get("failure_type", {}).value
                    if failure_info.get("failure_type")
                    else "unknown"
                ),
                failure_count=self.failure_count,
            )

            # If switching to safe mode or shutdown, pause primary strategy
            if new_mode in [FallbackMode.SAFE_MODE, FallbackMode.SHUTDOWN]:
                if self.primary_strategy:
                    await self.primary_strategy.pause()

        except Exception as e:
            self.logger.error("Error switching fallback mode", error=str(e))

    async def _check_recovery_conditions(self) -> None:
        """Check if conditions are suitable for recovery to primary strategy."""
        try:
            current_time = datetime.now(timezone.utc)

            # Must be in fallback mode for minimum duration
            if current_time - self.last_mode_change < timedelta(minutes=15):
                return

            # Check if market conditions have improved
            if len(self.market_data_history) >= 20:
                market_failure = self.failure_detector.detect_market_conditions(
                    self.market_data_history
                )
                if market_failure["detected"]:
                    return  # Market conditions still adverse

            # Check recent performance if in recovery mode
            if self.current_mode == FallbackMode.RECOVERY:
                if current_time - self.last_mode_change > self.recovery_test_duration:
                    # Evaluate recovery performance
                    recovery_success = await self._evaluate_recovery_performance()
                    if recovery_success:
                        await self._switch_mode(
                            FallbackMode.PRIMARY, {"reason": "recovery_successful"}
                        )
                        self.failure_count = 0  # Reset failure count
                        self.recovery_attempts = 0
                    else:
                        # Recovery failed, go back to fallback mode
                        fallback_mode = (
                            FallbackMode.DEGRADED
                            if self.failure_count < 2
                            else FallbackMode.SAFE_MODE
                        )
                        await self._switch_mode(fallback_mode, {"reason": "recovery_failed"})
                        self.recovery_attempts += 1
                return

            # Check if we can attempt recovery
            if self.recovery_attempts < 3:  # Limit recovery attempts
                await self._switch_mode(FallbackMode.RECOVERY, {"reason": "attempting_recovery"})

        except Exception as e:
            self.logger.error("Error checking recovery conditions", error=str(e))

    async def _evaluate_recovery_performance(self) -> bool:
        """Evaluate if recovery attempt was successful."""
        try:
            # Look at recent trades during recovery period
            recovery_start = self.last_mode_change
            recent_trades = [
                trade
                for trade in self.failure_detector.recent_trades
                if trade["timestamp"] > recovery_start
            ]

            if len(recent_trades) < 3:
                return False  # Not enough data

            # Calculate recovery metrics
            win_rate = sum(1 for trade in recent_trades if trade["is_win"]) / len(recent_trades)
            returns = [trade["return"] for trade in recent_trades]
            total_return = sum(returns)

            # Recovery success criteria
            if (
                win_rate >= self.recovery_performance_threshold
                and total_return > 0
                and len(recent_trades) >= 3
            ):
                return True

            return False

        except Exception as e:
            self.logger.error("Error evaluating recovery performance", error=str(e))
            return False

    async def _generate_signal_for_current_mode(self, data: MarketData) -> Signal | None:
        """Generate signal based on current fallback mode."""
        try:
            symbol = data.symbol
            price_history = self.price_history.get(symbol, [])

            if self.current_mode == FallbackMode.PRIMARY:
                # Use primary strategy
                if self.primary_strategy:
                    signals = await self.primary_strategy.generate_signals(data)
                    return signals[0] if signals else None
                else:
                    # No primary strategy, use degraded mode
                    return await self.degraded_mode_strategy.generate_signal(data, price_history)

            elif self.current_mode == FallbackMode.DEGRADED:
                return await self.degraded_mode_strategy.generate_signal(data, price_history)

            elif self.current_mode == FallbackMode.SAFE_MODE:
                return await self.safe_mode_strategy.generate_signal(data, price_history)

            elif self.current_mode == FallbackMode.RECOVERY:
                # Use primary strategy but with enhanced monitoring
                if self.primary_strategy:
                    signals = await self.primary_strategy.generate_signals(data)
                    if signals:
                        # Add recovery mode metadata
                        signal = signals[0]
                        if not signal.metadata:
                            signal.metadata = {}
                        signal.metadata["recovery_mode"] = True
                        return signal
                return None

            elif self.current_mode == FallbackMode.SHUTDOWN:
                return None  # No trading in shutdown mode

            return None

        except Exception as e:
            self.logger.error(
                "Error generating signal for current mode",
                mode=self.current_mode.value,
                error=str(e),
            )
            # Record error and potentially switch to safe mode
            self.failure_detector.add_error("signal_generation", datetime.now(timezone.utc))
            if self.current_mode != FallbackMode.SAFE_MODE:
                await self._switch_mode(
                    FallbackMode.SAFE_MODE,
                    {
                        "failure_type": FailureType.TECHNICAL_ERROR,
                        "reason": "signal_generation_error",
                    },
                )
            return None

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate signal with mode-specific criteria."""
        try:
            # Basic validation
            if signal.direction == SignalDirection.HOLD:
                return False

            # Mode-specific validation
            if self.current_mode == FallbackMode.SAFE_MODE:
                # Strict validation for safe mode
                return (
                    signal.confidence >= 0.8
                    and signal.metadata
                    and signal.metadata.get("mode") == "safe_mode"
                )

            elif self.current_mode == FallbackMode.DEGRADED:
                # Moderate validation for degraded mode
                return (
                    signal.confidence >= 0.6
                    and signal.metadata
                    and signal.metadata.get("mode") == "degraded_mode"
                )

            elif self.current_mode == FallbackMode.RECOVERY:
                # Enhanced validation for recovery mode
                return (
                    signal.confidence >= 0.7
                    and signal.metadata
                    and signal.metadata.get("recovery_mode")
                )

            else:
                # Standard validation for primary mode
                return signal.confidence >= self.config.min_confidence

        except Exception as e:
            self.logger.error("Error validating fallback signal", error=str(e))
            return False

    def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size based on current fallback mode."""
        try:
            base_size = Decimal(str(self.config.position_size_pct))

            # Apply mode-specific size adjustments
            if self.current_mode == FallbackMode.SAFE_MODE:
                size_multiplier = Decimal("0.5")  # 50% of normal size
            elif self.current_mode == FallbackMode.DEGRADED:
                size_multiplier = Decimal("0.7")  # 70% of normal size
            elif self.current_mode == FallbackMode.RECOVERY:
                size_multiplier = Decimal("0.8")  # 80% of normal size for testing
            else:
                size_multiplier = Decimal("1.0")  # Normal size

            # Adjust based on confidence
            confidence_multiplier = Decimal(str(signal.confidence))

            # Adjust based on failure count
            failure_adjustment = max(
                Decimal("0.5"), Decimal("1.0") - Decimal(str(self.failure_count * 0.1))
            )

            final_size = base_size * size_multiplier * confidence_multiplier * failure_adjustment

            # Apply mode-specific limits
            if self.current_mode == FallbackMode.SAFE_MODE:
                max_size = Decimal("0.01")  # 1% maximum in safe mode
            elif self.current_mode == FallbackMode.DEGRADED:
                max_size = Decimal("0.02")  # 2% maximum in degraded mode
            else:
                max_size = Decimal("0.05")  # 5% maximum normally

            min_size = Decimal("0.005")  # 0.5% minimum

            return max(min_size, min(max_size, final_size))

        except Exception as e:
            self.logger.error("Error calculating fallback position size", error=str(e))
            return Decimal("0.005")  # Conservative default

    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Determine exit with mode-specific logic."""
        try:
            # Basic stop-loss and take-profit
            current_price = data.current_price
            entry_price = position.entry_price

            if position.side.value == "buy":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # Mode-specific exit criteria
            if self.current_mode == FallbackMode.SAFE_MODE:
                # Tighter stops in safe mode
                stop_loss = self.config.stop_loss_pct * 0.5  # 50% tighter
                take_profit = self.config.take_profit_pct * 0.7  # Earlier profit taking
            elif self.current_mode == FallbackMode.DEGRADED:
                # Slightly tighter stops
                stop_loss = self.config.stop_loss_pct * 0.8
                take_profit = self.config.take_profit_pct * 0.9
            else:
                # Normal stops
                stop_loss = self.config.stop_loss_pct
                take_profit = self.config.take_profit_pct

            # Check stops
            if pnl_pct <= -stop_loss:
                self.logger.info(
                    "Stop loss triggered", symbol=position.symbol, mode=self.current_mode.value
                )
                return True

            if pnl_pct >= take_profit:
                self.logger.info(
                    "Take profit triggered", symbol=position.symbol, mode=self.current_mode.value
                )
                return True

            # Emergency exit in safe mode if any loss
            if self.current_mode == FallbackMode.SAFE_MODE and pnl_pct < -0.005:  # 0.5% loss
                self.logger.info("Emergency exit in safe mode", symbol=position.symbol)
                return True

            return False

        except Exception as e:
            self.logger.error(
                "Error in fallback exit decision", symbol=position.symbol, error=str(e)
            )
            return True  # Exit on error for safety

    def update_trade_result(self, return_pct: float, trade_info: dict[str, Any]) -> None:
        """Update trade result for failure detection."""
        try:
            timestamp = trade_info.get("timestamp", datetime.now(timezone.utc))
            self.failure_detector.add_trade_result(return_pct, timestamp)

            self.logger.debug(
                "Trade result updated",
                return_pct=return_pct,
                mode=self.current_mode.value,
                failure_count=self.failure_count,
            )

        except Exception as e:
            self.logger.error("Error updating trade result", error=str(e))

    def get_fallback_statistics(self) -> dict[str, Any]:
        """Get comprehensive fallback statistics."""
        try:
            return {
                "current_mode": self.current_mode.value,
                "failure_count": self.failure_count,
                "recovery_attempts": self.recovery_attempts,
                "last_mode_change": self.last_mode_change.isoformat(),
                "mode_history": [
                    {**history, "timestamp": history["timestamp"].isoformat()}
                    for history in self.mode_history[-10:]  # Last 10 mode changes
                ],
                "primary_strategy": self.primary_strategy.name if self.primary_strategy else None,
                "failure_detector_stats": {
                    "recent_trades": len(self.failure_detector.recent_trades),
                    "recent_errors": len(self.failure_detector.recent_errors),
                    "recent_timeouts": len(self.failure_detector.recent_timeouts),
                    "recent_signals": len(self.failure_detector.last_signals),
                },
                "performance_thresholds": {
                    "max_drawdown_threshold": self.failure_detector.max_drawdown_threshold,
                    "min_win_rate_threshold": self.failure_detector.min_win_rate_threshold,
                    "consecutive_loss_threshold": self.failure_detector.consecutive_loss_threshold,
                },
            }

        except Exception as e:
            self.logger.error("Error getting fallback statistics", error=str(e))
            return {"error": str(e)}
