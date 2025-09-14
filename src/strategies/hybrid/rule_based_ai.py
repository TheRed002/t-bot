"""
Rule-Based AI Hybrid Strategy

This module implements a hybrid strategy that combines traditional technical analysis
rules with AI predictions. It provides sophisticated conflict resolution between
rule-based and AI-driven decisions, with dynamic weight adjustment based on performance.

Key Features:
- Traditional technical indicator rules
- AI prediction integration with confidence weighting
- Rule validation against AI predictions
- Performance attribution between different components
- Dynamic weight adjustment based on historical performance
- Conflict resolution mechanisms
"""

from decimal import Decimal
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from src.core.exceptions import StrategyError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import MarketData, Position, PositionSide, Signal, SignalDirection, StrategyType

# MANDATORY: Import from P-010
from src.risk_management.regime_detection import MarketRegimeDetector

# MANDATORY: Import from P-011
from src.strategies.base import BaseStrategy
from src.strategies.dependencies import StrategyServiceContainer

# MANDATORY: Import from P-007A
from src.utils.decorators import time_execution


class TechnicalRuleEngine:
    """Traditional technical analysis rule engine.
    
    Uses shared BaseStrategy technical indicator methods to eliminate code duplication.
    """

    def __init__(self, config: dict[str, Any], strategy_instance: Optional["BaseStrategy"] = None):
        """Initialize the technical rule engine.
        
        Args:
            config: Configuration parameters
            strategy_instance: Reference to BaseStrategy instance for shared indicator methods
        """
        self.config = config
        self.strategy_instance = strategy_instance  # Reference for shared methods
        self.rsi_period = config.get("rsi_period", 14)
        self.ma_short_period = config.get("ma_short_period", 10)
        self.ma_long_period = config.get("ma_long_period", 20)
        self.volume_threshold = config.get("volume_threshold", 1.5)

        # Rule weights (can be adjusted dynamically)
        self.rule_weights = {
            "rsi_oversold": 0.3,
            "rsi_overbought": 0.3,
            "ma_crossover": 0.4,
            "volume_confirmation": 0.2,
            "trend_strength": 0.3,
        }

        # Performance tracking for each rule
        self.rule_performance = {
            rule: {"wins": 0, "losses": 0, "total": 0} for rule in self.rule_weights.keys()
        }

    async def calculate_rsi(self, symbol: str) -> float:
        """Calculate RSI indicator using shared BaseStrategy method.
        
        This eliminates code duplication by using the BaseStrategy's shared RSI calculation.
        """
        if not self.strategy_instance:
            return 50.0  # Neutral RSI if no strategy instance

        try:
            # Use shared BaseStrategy method for RSI calculation
            rsi_decimal = await self.strategy_instance.get_rsi(symbol, self.rsi_period)

            if rsi_decimal is not None:
                return rsi_decimal
            else:
                return 50.0  # Neutral RSI on calculation failure
        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            return 50.0

    async def calculate_moving_averages(self, symbol: str, current_price: Decimal) -> tuple[Decimal, Decimal]:
        """Calculate short and long moving averages using shared BaseStrategy methods.
        
        This eliminates code duplication by using the BaseStrategy's shared SMA calculations.
        """
        if not self.strategy_instance:
            return current_price, current_price  # Fallback if no strategy instance

        try:
            # Use shared BaseStrategy methods for SMA calculations
            ma_short_decimal = await self.strategy_instance.get_sma(symbol, self.ma_short_period)
            ma_long_decimal = await self.strategy_instance.get_sma(symbol, self.ma_long_period)

            ma_short = ma_short_decimal if ma_short_decimal is not None else current_price
            ma_long = ma_long_decimal if ma_long_decimal is not None else current_price

            return ma_short, ma_long
        except Exception as e:
            logger.error(f"Failed to calculate moving averages: {e}")
            return current_price, current_price

    async def evaluate_rules(
        self, market_data: MarketData, price_history: list[float], volume_history: list[float]
    ) -> dict[str, Any]:
        """Evaluate all technical rules and return signals.
        
        Updated to use shared BaseStrategy methods for technical indicators.
        """
        signals = []
        rules_triggered = []

        # RSI Rules - using shared BaseStrategy method
        rsi = await self.calculate_rsi(market_data.symbol)
        if rsi < 30:  # Oversold
            signals.append(
                {
                    "rule": "rsi_oversold",
                    "direction": SignalDirection.BUY,
                    "confidence": min((30 - rsi) / 10, 1.0),
                    "weight": self.rule_weights["rsi_oversold"],
                }
            )
            rules_triggered.append("rsi_oversold")
        elif rsi > 70:  # Overbought
            signals.append(
                {
                    "rule": "rsi_overbought",
                    "direction": SignalDirection.SELL,
                    "confidence": min((rsi - 70) / 10, 1.0),
                    "weight": self.rule_weights["rsi_overbought"],
                }
            )
            rules_triggered.append("rsi_overbought")

        # Moving Average Crossover - using shared BaseStrategy methods
        current_price = market_data.price if isinstance(market_data.price, Decimal) else Decimal(str(market_data.price))
        ma_short, ma_long = await self.calculate_moving_averages(market_data.symbol, current_price)
        if ma_short > ma_long * Decimal("1.005"):  # 0.5% threshold for noise filtering
            crossover_strength = (ma_short - ma_long) / ma_long
            signals.append(
                {
                    "rule": "ma_crossover",
                    "direction": SignalDirection.BUY,
                    "confidence": min(crossover_strength * 10, 1.0),
                    "weight": self.rule_weights["ma_crossover"],
                }
            )
            rules_triggered.append("ma_crossover_bullish")
        elif ma_short < ma_long * 0.995:
            crossover_strength = (ma_long - ma_short) / ma_long
            signals.append(
                {
                    "rule": "ma_crossover",
                    "direction": SignalDirection.SELL,
                    "confidence": min(crossover_strength * 10, 1.0),
                    "weight": self.rule_weights["ma_crossover"],
                }
            )
            rules_triggered.append("ma_crossover_bearish")

        # Volume confirmation
        if len(volume_history) >= 5:
            avg_volume = np.mean(volume_history[-5:])
            current_volume = market_data.volume if isinstance(market_data.volume, Decimal) else Decimal(str(market_data.volume))
            if current_volume > avg_volume * self.volume_threshold:
                signals.append(
                    {
                        "rule": "volume_confirmation",
                        "direction": (
                            SignalDirection.BUY
                            if signals and signals[-1]["direction"] == SignalDirection.BUY
                            else SignalDirection.SELL
                        ),
                        "confidence": min((current_volume / avg_volume - 1), 1.0),
                        "weight": self.rule_weights["volume_confirmation"],
                    }
                )
                rules_triggered.append("high_volume")

        return {
            "signals": signals,
            "rules_triggered": rules_triggered,
            "rsi": rsi,
            "ma_short": ma_short,
            "ma_long": ma_long,
        }

    def update_rule_performance(self, rule: str, performance: float) -> None:
        """Update performance tracking for a specific rule."""
        if rule in self.rule_performance:
            self.rule_performance[rule]["total"] += 1
            if performance > 0:
                self.rule_performance[rule]["wins"] += 1
            else:
                self.rule_performance[rule]["losses"] += 1

    def adjust_rule_weights(self) -> None:
        """Dynamically adjust rule weights based on performance."""
        for rule, performance in self.rule_performance.items():
            if performance["total"] > 10:  # Minimum trades for adjustment
                win_rate = performance["wins"] / performance["total"]
                # Adjust weight based on win rate (0.5 = neutral)
                adjustment = (win_rate - 0.5) * 0.2  # Max 10% adjustment
                self.rule_weights[rule] = max(0.1, min(1.0, self.rule_weights[rule] + adjustment))


class AIPredictor:
    """AI prediction component using machine learning."""

    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None):
        """Initialize AI predictor."""
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 10),
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_window = config.get("feature_window", 50)
        self.retrain_frequency = config.get("retrain_frequency", 100)
        self.prediction_count = 0
        self.logger = get_logger(__name__)

        # Performance tracking
        self.predictions_made: list[Any] = []
        self.actual_outcomes: list[Any] = []

    def prepare_features(
        self, price_history: list[float], volume_history: list[float]
    ) -> np.ndarray:
        """Prepare features for ML model."""
        if len(price_history) < self.feature_window:
            return np.array([])

        features = []

        # Price-based features
        returns = np.diff(np.log(price_history[-self.feature_window :]))
        features.extend(
            [
                np.mean(returns),  # Average return
                np.std(returns),  # Volatility
                np.percentile(returns, 25),  # 25th percentile
                np.percentile(returns, 75),  # 75th percentile
            ]
        )

        # Technical indicators as features
        prices = np.array(price_history[-self.feature_window :])

        # Moving averages
        ma_5 = np.mean(prices[-5:])
        ma_10 = np.mean(prices[-10:])
        ma_20 = np.mean(prices[-20:])

        features.extend(
            [
                (ma_5 - ma_10) / ma_10,  # MA ratio 5/10
                (ma_10 - ma_20) / ma_20,  # MA ratio 10/20
            ]
        )

        # RSI
        if len(price_history) >= 14:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
            features.append(rsi / 100)  # Normalize RSI
        else:
            features.append(0.5)  # Neutral RSI

        # Volume features
        if len(volume_history) >= 10:
            volumes = np.array(volume_history[-10:])
            features.extend(
                [
                    np.mean(volumes),
                    np.std(volumes),
                    volumes[-1] / np.mean(volumes[:-1]),  # Current vs average volume
                ]
            )
        else:
            features.extend([0, 0, 1])  # Default volume features

        return np.array(features).reshape(1, -1)

    async def train_model(self, training_data: list[dict[str, Any]]) -> None:
        """Train the ML model with historical data."""
        if len(training_data) < 50:
            self.logger.warning("Insufficient training data for AI model")
            return

        try:
            X = []
            y = []

            for data_point in training_data:
                features = self.prepare_features(
                    data_point["price_history"], data_point["volume_history"]
                )
                if features.size > 0:
                    X.append(features.flatten())
                    # Label: 1 for price increase, 0 for decrease
                    y.append(1 if data_point["future_return"] > 0 else 0)

            if len(X) < 10:
                self.logger.warning("Insufficient valid features for training")
                return

            X = np.array(X)
            y = np.array(y)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True

            # Calculate training accuracy
            y_pred = self.model.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred)

            self.logger.info(
                "AI model trained successfully",
                training_samples=len(X),
                accuracy=accuracy,
                features=X.shape[1],
            )

        except Exception as e:
            self.logger.error("Error training AI model", error=str(e))
            raise StrategyError(f"AI model training failed: {e}")

    async def predict(
        self, price_history: list[float], volume_history: list[float]
    ) -> dict[str, Any]:
        """Make AI prediction."""
        if not self.is_trained:
            return {
                "direction": SignalDirection.HOLD,
                "confidence": 0.0,
                "probabilities": [0.5, 0.5],
            }

        try:
            features = self.prepare_features(price_history, volume_history)
            if features.size == 0:
                return {
                    "direction": SignalDirection.HOLD,
                    "confidence": 0.0,
                    "probabilities": [0.5, 0.5],
                }

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Get prediction probabilities
            probabilities = self.model.predict_proba(features_scaled)[0]
            prediction = self.model.predict(features_scaled)[0]

            # Convert to trading signal
            if prediction == 1 and probabilities[1] > 0.6:
                direction = SignalDirection.BUY
                confidence = probabilities[1]
            elif prediction == 0 and probabilities[0] > 0.6:
                direction = SignalDirection.SELL
                confidence = probabilities[0]
            else:
                direction = SignalDirection.HOLD
                confidence = max(probabilities)

            self.prediction_count += 1

            return {
                "direction": direction,
                "confidence": confidence,
                "probabilities": probabilities.tolist(),
            }

        except Exception as e:
            self.logger.error("Error making AI prediction", error=str(e))
            return {
                "direction": SignalDirection.HOLD,
                "confidence": 0.0,
                "probabilities": [0.5, 0.5],
            }

    def update_performance(self, prediction: dict[str, Any], actual_outcome: float) -> None:
        """Update AI performance tracking."""
        self.predictions_made.append(prediction)
        self.actual_outcomes.append(actual_outcome)

        # Keep only recent predictions for performance calculation
        if len(self.predictions_made) > 1000:
            self.predictions_made = self.predictions_made[-1000:]
            self.actual_outcomes = self.actual_outcomes[-1000:]

    def get_performance_metrics(self) -> dict[str, float]:
        """Get AI performance metrics."""
        if len(self.predictions_made) < 10:
            return {"accuracy": 0.5, "precision": 0.5, "recall": 0.5}

        correct_predictions = 0
        total_predictions = len(self.predictions_made)

        for pred, outcome in zip(self.predictions_made, self.actual_outcomes, strict=False):
            predicted_direction = pred["direction"]
            actual_direction = SignalDirection.BUY if outcome > 0 else SignalDirection.SELL

            if (
                predicted_direction == actual_direction
                or predicted_direction == SignalDirection.HOLD
            ):
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        return {"accuracy": accuracy, "precision": accuracy, "recall": accuracy}


class RuleBasedAIStrategy(BaseStrategy):
    """
    Hybrid strategy combining traditional technical analysis rules with AI predictions.

    This strategy implements sophisticated conflict resolution between rule-based and
    AI-driven decisions, with dynamic weight adjustment based on performance attribution.
    """

    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None):
        """Initialize the Rule-Based AI hybrid strategy."""
        # Ensure strategy type is set correctly
        config["strategy_type"] = StrategyType.CUSTOM
        if "name" not in config:
            config["name"] = "RuleBasedAI"

        super().__init__(config, services)

        # Initialize components with strategy instance reference for shared indicator methods
        self.rule_engine = TechnicalRuleEngine(config.get("rules", {}), strategy_instance=self)
        self.ai_predictor = AIPredictor(config.get("ai", {}))
        self.regime_detector = MarketRegimeDetector(config.get("regime_detection", {}))

        # Initialize ML service integration
        self._ml_service = None
        if services and services.ml_service:
            self._ml_service = services.ml_service
            self.logger.info("ML service integrated for enhanced AI predictions", strategy=self.name)

        # Hybrid strategy parameters
        self.rule_weight = config.get("rule_weight", 0.6)
        self.ai_weight = config.get("ai_weight", 0.4)
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.65)
        self.conflict_resolution_method = config.get("conflict_resolution", "weighted_average")

        # Data storage for analysis
        self.price_history: dict[str, list[float]] = {}
        self.volume_history: dict[str, list[float]] = {}
        self.signal_history: list[dict[str, Any]] = []

        # Performance attribution
        self.component_performance = {
            "rules": {"wins": 0, "losses": 0, "total": 0},
            "ai": {"wins": 0, "losses": 0, "total": 0},
            "hybrid": {"wins": 0, "losses": 0, "total": 0},
        }

        self.logger.info(
            "RuleBasedAI strategy initialized",
            rule_weight=self.rule_weight,
            ai_weight=self.ai_weight,
            conflict_resolution=self.conflict_resolution_method,
        )

    @property
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        return StrategyType.CUSTOM

    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """Generate hybrid signals combining rules and AI predictions."""
        try:
            symbol = data.symbol

            # Update price and volume history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                self.volume_history[symbol] = []

            self.price_history[symbol].append(float(data.price if isinstance(data.price, Decimal) else Decimal(str(data.price))))
            self.volume_history[symbol].append(float(data.volume if isinstance(data.volume, Decimal) else Decimal(str(data.volume))))

            # Keep only recent history (last 200 points)
            self.price_history[symbol] = self.price_history[symbol][-200:]
            self.volume_history[symbol] = self.volume_history[symbol][-200:]

            # Need sufficient data for analysis
            if len(self.price_history[symbol]) < 50:
                self.logger.debug("Insufficient price history for signal generation", symbol=symbol)
                return []

            # Get market regime
            current_regime = await self.regime_detector.detect_comprehensive_regime([data])

            # Generate rule-based signals
            rule_analysis = await self.rule_engine.evaluate_rules(
                data, self.price_history[symbol], self.volume_history[symbol]
            )

            # Generate AI prediction using both local predictor and ML service
            ai_prediction = await self.ai_predictor.predict(
                self.price_history[symbol], self.volume_history[symbol]
            )

            # Enhanced prediction using ML service if available
            if self._ml_service:
                try:
                    enhanced_prediction = await self._get_ml_service_prediction(symbol, data)
                    if enhanced_prediction:
                        # Combine predictions with confidence weighting
                        ai_prediction = self._combine_predictions(ai_prediction, enhanced_prediction)
                except Exception as e:
                    self.logger.warning(f"ML service prediction failed: {e}", symbol=symbol)

            # Combine signals using conflict resolution
            hybrid_signal = await self._resolve_conflicts(
                rule_analysis, ai_prediction, current_regime, data
            )

            if hybrid_signal:
                # Store signal for performance tracking
                signal_record = {
                    "timestamp": data.timestamp,
                    "symbol": symbol,
                    "rule_analysis": rule_analysis,
                    "ai_prediction": ai_prediction,
                    "hybrid_signal": hybrid_signal.model_dump(),
                    "regime": current_regime.value,
                }
                self.signal_history.append(signal_record)

                # Keep only recent signals
                if len(self.signal_history) > 1000:
                    self.signal_history = self.signal_history[-1000:]

                return [hybrid_signal]

            return []

        except Exception as e:
            self.logger.error("Error generating hybrid signals", symbol=data.symbol, error=str(e))
            return []

    async def _resolve_conflicts(
        self,
        rule_analysis: dict[str, Any],
        ai_prediction: dict[str, Any],
        regime: Any,
        data: MarketData,
    ) -> Signal | None:
        """Resolve conflicts between rule-based and AI predictions."""
        try:
            # Extract rule-based signals
            rule_signals = rule_analysis.get("signals", [])
            if not rule_signals:
                rule_direction = SignalDirection.HOLD
                rule_confidence = 0.0
            else:
                # Combine multiple rule signals
                buy_weight = sum(
                    s["confidence"] * s["weight"]
                    for s in rule_signals
                    if s["direction"] == SignalDirection.BUY
                )
                sell_weight = sum(
                    s["confidence"] * s["weight"]
                    for s in rule_signals
                    if s["direction"] == SignalDirection.SELL
                )

                if buy_weight > sell_weight and buy_weight > 0.3:
                    rule_direction = SignalDirection.BUY
                    rule_confidence = min(buy_weight, 1.0)
                elif sell_weight > buy_weight and sell_weight > 0.3:
                    rule_direction = SignalDirection.SELL
                    rule_confidence = min(sell_weight, 1.0)
                else:
                    rule_direction = SignalDirection.HOLD
                    rule_confidence = 0.0

            # Get AI prediction
            ai_direction = ai_prediction.get("direction", SignalDirection.HOLD)
            ai_confidence = ai_prediction.get("confidence", 0.0)

            # Apply conflict resolution method
            if self.conflict_resolution_method == "weighted_average":
                final_signal = self._weighted_average_resolution(
                    rule_direction, rule_confidence, ai_direction, ai_confidence
                )
            elif self.conflict_resolution_method == "highest_confidence":
                final_signal = self._highest_confidence_resolution(
                    rule_direction, rule_confidence, ai_direction, ai_confidence
                )
            elif self.conflict_resolution_method == "consensus_required":
                final_signal = self._consensus_resolution(
                    rule_direction, rule_confidence, ai_direction, ai_confidence
                )
            else:
                # Default to weighted average
                final_signal = self._weighted_average_resolution(
                    rule_direction, rule_confidence, ai_direction, ai_confidence
                )

            # Check minimum confidence threshold
            if final_signal and final_signal["confidence"] >= self.min_confidence_threshold:
                return Signal(
                    symbol=data.symbol,
                    direction=final_signal["direction"],
                    strength=Decimal(str(final_signal["confidence"])),
                    timestamp=data.timestamp,
                    source=self.name,
                    metadata={
                        "rule_direction": rule_direction.value,
                        "rule_confidence": rule_confidence,
                        "ai_direction": ai_direction.value,
                        "ai_confidence": ai_confidence,
                        "resolution_method": self.conflict_resolution_method,
                        "regime": regime.value,
                        "rule_analysis": rule_analysis,
                        "ai_probabilities": ai_prediction.get("probabilities", []),
                    },
                )

            return None

        except Exception as e:
            self.logger.error("Error resolving signal conflicts", error=str(e))
            return None

    def _weighted_average_resolution(
        self,
        rule_direction: SignalDirection,
        rule_confidence: float,
        ai_direction: SignalDirection,
        ai_confidence: float,
    ) -> dict[str, Any] | None:
        """Resolve conflicts using weighted average approach."""
        # Calculate weighted scores
        rule_score = 0
        ai_score = 0

        if rule_direction == SignalDirection.BUY:
            rule_score = rule_confidence * self.rule_weight
        elif rule_direction == SignalDirection.SELL:
            rule_score = -rule_confidence * self.rule_weight

        if ai_direction == SignalDirection.BUY:
            ai_score = ai_confidence * self.ai_weight
        elif ai_direction == SignalDirection.SELL:
            ai_score = -ai_confidence * self.ai_weight

        combined_score = rule_score + ai_score
        combined_confidence = abs(combined_score)

        if combined_score > 0.1:
            direction = SignalDirection.BUY
        elif combined_score < -0.1:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD

        return {"direction": direction, "confidence": combined_confidence}

    def _highest_confidence_resolution(
        self,
        rule_direction: SignalDirection,
        rule_confidence: float,
        ai_direction: SignalDirection,
        ai_confidence: float,
    ) -> dict[str, Any] | None:
        """Use the prediction with highest confidence."""
        if rule_confidence > ai_confidence:
            return {"direction": rule_direction, "confidence": rule_confidence}
        else:
            return {"direction": ai_direction, "confidence": ai_confidence}

    def _consensus_resolution(
        self,
        rule_direction: SignalDirection,
        rule_confidence: float,
        ai_direction: SignalDirection,
        ai_confidence: float,
    ) -> dict[str, Any] | None:
        """Require consensus between rules and AI."""
        if rule_direction == ai_direction and rule_direction != SignalDirection.HOLD:
            # Both agree on direction
            combined_confidence = (rule_confidence + ai_confidence) / 2
            return {"direction": rule_direction, "confidence": combined_confidence}
        else:
            # No consensus - hold
            return {"direction": SignalDirection.HOLD, "confidence": 0.0}

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate hybrid signal before execution."""
        try:
            # Basic signal validation
            if signal.strength < Decimal(str(self.min_confidence_threshold)):
                self.logger.warning("Signal strength below threshold", strength=signal.strength)
                return False

            if signal.direction == SignalDirection.HOLD:
                return False

            # Validate metadata exists
            if not signal.metadata:
                self.logger.warning("Signal missing metadata")
                return False

            # Check for required metadata fields
            required_fields = ["rule_direction", "ai_direction", "resolution_method"]
            for field in required_fields:
                if field not in signal.metadata:
                    self.logger.warning("Signal missing required metadata field", field=field)
                    return False

            return True

        except Exception as e:
            self.logger.error("Error validating hybrid signal", error=str(e))
            return False

    def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size based on signal confidence and component agreement."""
        try:
            base_size = Decimal(str(self.config.position_size_pct))

            # Adjust size based on confidence
            confidence_multiplier = signal.strength

            # Adjust based on component agreement
            rule_direction = signal.metadata.get("rule_direction", "hold")
            ai_direction = signal.metadata.get("ai_direction", "hold")

            agreement_multiplier = Decimal("1.0")
            if rule_direction == ai_direction and rule_direction != "hold":
                # Both components agree - increase position size
                agreement_multiplier = Decimal("1.2")
            elif rule_direction != ai_direction:
                # Disagreement - reduce position size
                agreement_multiplier = Decimal("0.8")

            final_size = base_size * confidence_multiplier * agreement_multiplier

            # Ensure size is within reasonable bounds
            max_size = Decimal("0.05")  # 5% maximum
            min_size = Decimal("0.005")  # 0.5% minimum

            return max(min_size, min(max_size, final_size))

        except Exception as e:
            self.logger.error("Error calculating position size", error=str(e))
            return Decimal("0.01")  # Default 1%

    async def should_exit(self, position: Position, data: MarketData) -> bool:
        """Determine if position should be closed using hybrid analysis."""
        try:
            # Basic stop-loss and take-profit checks
            current_price = data.price
            entry_price = position.entry_price

            if position.side == PositionSide.LONG:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # Stop loss
            if pnl_pct <= -self.config.stop_loss_pct:
                self.logger.info("Stop loss triggered", symbol=position.symbol, pnl_pct=pnl_pct)
                return True

            # Take profit
            if pnl_pct >= self.config.take_profit_pct:
                self.logger.info("Take profit triggered", symbol=position.symbol, pnl_pct=pnl_pct)
                return True

            # Additional hybrid-based exit logic
            symbol = position.symbol
            if symbol in self.price_history and len(self.price_history[symbol]) >= 20:
                # Get current rule and AI signals for exit decision
                rule_analysis = await self.rule_engine.evaluate_rules(
                    data, self.price_history[symbol], self.volume_history.get(symbol, [])
                )

                # Check for strong opposing signals
                rule_signals = rule_analysis.get("signals", [])
                opposing_signals = [
                    s
                    for s in rule_signals
                    if (position.side.value == "buy" and s["direction"] == SignalDirection.SELL)
                    or (position.side.value == "sell" and s["direction"] == SignalDirection.BUY)
                ]

                if opposing_signals:
                    opposing_strength = sum(s["confidence"] * s["weight"] for s in opposing_signals)
                    if opposing_strength > 0.6:  # Strong opposing signal
                        self.logger.info(
                            "Hybrid exit signal triggered",
                            symbol=position.symbol,
                            opposing_strength=opposing_strength,
                        )
                        return True

            return False

        except Exception as e:
            self.logger.error("Error in exit decision", symbol=position.symbol, error=str(e))
            return False

    async def _on_start(self) -> None:
        """Initialize strategy components on start."""
        self.logger.info("Starting RuleBasedAI hybrid strategy")

        # If we have historical data, train the AI model
        if len(self.signal_history) > 50:
            await self._retrain_ai_model()

    async def _retrain_ai_model(self) -> None:
        """Retrain AI model with recent performance data."""
        try:
            training_data = []

            for i, signal_record in enumerate(self.signal_history):
                if i < len(self.signal_history) - 10:  # Need future data for labeling
                    # Calculate future return for labeling
                    current_price = signal_record["hybrid_signal"]["metadata"]["rule_analysis"][
                        "ma_short"
                    ]
                    future_signals = self.signal_history[i + 1 : i + 11]  # Next 10 signals
                    if future_signals:
                        future_price = future_signals[-1]["hybrid_signal"]["metadata"][
                            "rule_analysis"
                        ]["ma_short"]
                        future_return = (future_price - current_price) / current_price

                        training_data.append(
                            {
                                "price_history": self.price_history.get(
                                    signal_record["symbol"], []
                                )[-50:],
                                "volume_history": self.volume_history.get(
                                    signal_record["symbol"], []
                                )[-50:],
                                "future_return": future_return,
                            }
                        )

            if len(training_data) > 20:
                await self.ai_predictor.train_model(training_data)
                self.logger.info("AI model retrained", training_samples=len(training_data))

        except Exception as e:
            self.logger.error("Error retraining AI model", error=str(e))

    def adjust_component_weights(self) -> None:
        """Dynamically adjust weights based on component performance."""
        try:
            if (
                self.component_performance["rules"]["total"] > 20
                and self.component_performance["ai"]["total"] > 20
            ):
                # Calculate win rates
                rule_win_rate = (
                    self.component_performance["rules"]["wins"]
                    / self.component_performance["rules"]["total"]
                )
                ai_win_rate = (
                    self.component_performance["ai"]["wins"]
                    / self.component_performance["ai"]["total"]
                )

                # Adjust weights based on relative performance
                total_performance = rule_win_rate + ai_win_rate
                if total_performance > 0:
                    new_rule_weight = rule_win_rate / total_performance
                    new_ai_weight = ai_win_rate / total_performance

                    # Smooth the adjustment (don't change too rapidly)
                    adjustment_rate = 0.1
                    self.rule_weight = (
                        self.rule_weight * (1 - adjustment_rate) + new_rule_weight * adjustment_rate
                    )
                    self.ai_weight = (
                        self.ai_weight * (1 - adjustment_rate) + new_ai_weight * adjustment_rate
                    )

                    # Normalize weights
                    total_weight = self.rule_weight + self.ai_weight
                    self.rule_weight /= total_weight
                    self.ai_weight /= total_weight

                    self.logger.info(
                        "Component weights adjusted",
                        rule_weight=self.rule_weight,
                        ai_weight=self.ai_weight,
                        rule_win_rate=rule_win_rate,
                        ai_win_rate=ai_win_rate,
                    )

        except Exception as e:
            self.logger.error("Error adjusting component weights", error=str(e))

    def get_strategy_statistics(self) -> dict[str, Any]:
        """Get strategy performance statistics."""
        return {
            "total_trades": len(self.trade_history),
            "win_rate": self.overall_win_rate,
            "rule_performance": self.rule_performance.copy(),
            "component_weights": {
                "technical": self.technical_weight,
                "ai": self.ai_weight,
            },
        }


    # Helper methods for accessing data through data service

    def get_strategy_stats(self) -> dict[str, Any]:
        """Get comprehensive strategy statistics."""
        stats = {
            "component_weights": {"rule_weight": self.rule_weight, "ai_weight": self.ai_weight},
            "component_performance": self.component_performance,
            "rule_performance": self.rule_engine.rule_performance,
            "ai_performance": self.ai_predictor.get_performance_metrics(),
            "signal_history_length": len(self.signal_history),
            "ai_model_trained": self.ai_predictor.is_trained,
            "current_regime": self.regime_detector.get_current_regime().value,
            "conflict_resolution_method": self.conflict_resolution_method,
        }

        return stats

    async def _get_ml_service_prediction(self, symbol: str, data: MarketData) -> dict[str, Any] | None:
        """Get prediction from ML service for enhanced AI analysis.
        
        Args:
            symbol: Trading symbol
            data: Current market data
            
        Returns:
            ML prediction with confidence score or None if unavailable
        """
        try:
            if not self._ml_service:
                return None

            # Prepare feature data for ML service
            features = {
                "symbol": symbol,
                "price": float(data.price if isinstance(data.price, Decimal) else Decimal(str(data.price))),
                "volume": float(data.volume if isinstance(data.volume, Decimal) else Decimal(str(data.volume))),
                "timestamp": data.timestamp.isoformat(),
                "price_history": self.price_history.get(symbol, [])[-20:],  # Last 20 prices
                "volume_history": self.volume_history.get(symbol, [])[-20:],  # Last 20 volumes
            }

            # Get prediction from ML service
            if hasattr(self._ml_service, "predict_price_direction"):
                prediction = await self._ml_service.predict_price_direction(features)
                if prediction:
                    return {
                        "direction": prediction.get("direction", "HOLD"),
                        "confidence": prediction.get("confidence", 0.5),
                        "probability": prediction.get("probability", 0.5),
                        "source": "ml_service"
                    }

            # Fallback to general prediction
            if hasattr(self._ml_service, "predict"):
                prediction = await self._ml_service.predict(symbol, features)
                if prediction:
                    return {
                        "direction": prediction.get("direction", "HOLD"),
                        "confidence": prediction.get("confidence", 0.5),
                        "probability": prediction.get("probability", 0.5),
                        "source": "ml_service"
                    }

            return None

        except Exception as e:
            self.logger.warning(f"Failed to get ML service prediction: {e}", symbol=symbol)
            return None

    def _combine_predictions(self, local_prediction: dict[str, Any], ml_prediction: dict[str, Any]) -> dict[str, Any]:
        """Combine local AI prediction with ML service prediction.
        
        Args:
            local_prediction: Prediction from local AI predictor
            ml_prediction: Prediction from ML service
            
        Returns:
            Combined prediction with enhanced confidence
        """
        try:
            local_confidence = local_prediction.get("confidence", 0.5)
            ml_confidence = ml_prediction.get("confidence", 0.5)

            # Weighted average based on confidence levels
            total_weight = local_confidence + ml_confidence
            if total_weight == 0:
                return local_prediction

            local_weight = local_confidence / total_weight
            ml_weight = ml_confidence / total_weight

            # Combine directions with confidence weighting
            local_direction = local_prediction.get("direction", "HOLD")
            ml_direction = ml_prediction.get("direction", "HOLD")

            # If predictions agree, increase confidence
            if local_direction == ml_direction:
                combined_confidence = min(0.95, local_confidence * 0.6 + ml_confidence * 0.4 + 0.1)
                combined_direction = local_direction
            else:
                # If predictions disagree, use confidence-weighted approach
                if local_confidence > ml_confidence:
                    combined_direction = local_direction
                    combined_confidence = local_confidence * 0.8  # Reduce confidence due to disagreement
                else:
                    combined_direction = ml_direction
                    combined_confidence = ml_confidence * 0.8

            # Combine probability scores
            local_probability = local_prediction.get("probability", 0.5)
            ml_probability = ml_prediction.get("probability", 0.5)
            combined_probability = local_probability * local_weight + ml_probability * ml_weight

            enhanced_prediction = {
                "direction": combined_direction,
                "confidence": combined_confidence,
                "probability": combined_probability,
                "source": "hybrid_ml",
                "components": {
                    "local": local_prediction,
                    "ml_service": ml_prediction
                }
            }

            self.logger.debug(
                "Combined ML predictions",
                strategy=self.name,
                local_direction=local_direction,
                ml_direction=ml_direction,
                combined_direction=combined_direction,
                combined_confidence=combined_confidence
            )

            return enhanced_prediction

        except Exception as e:
            self.logger.warning(f"Failed to combine predictions: {e}")
            return local_prediction  # Fallback to local prediction
