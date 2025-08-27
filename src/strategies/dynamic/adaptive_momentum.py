"""
Adaptive Momentum Strategy Implementation - REFACTORED Day 13

This module implements a regime-aware momentum strategy using the enhanced
BaseStrategy architecture with service layer integration, centralized indicator
calculations, and state persistence.

Refactoring improvements:
- Uses StrategyService for lifecycle management
- Uses TechnicalIndicators service for centralized calculations
- Implements proper state persistence
- Enhanced metrics tracking and monitoring
- Removed direct data access patterns

CRITICAL: This strategy follows the perfect architecture from Day 12.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np

# MANDATORY: Import from P-001 - Core types
from src.core.types import (
    MarketData,
    MarketRegime,
    Position,
    Signal,
    SignalDirection,
    StrategyType,
)

# Import service layer components
from src.data.features.technical_indicators import TechnicalIndicators
from src.risk_management.adaptive_risk import AdaptiveRiskManager
from src.risk_management.regime_detection import MarketRegimeDetector
from src.strategies.base import BaseStrategy
from src.strategies.service import StrategyService

# MANDATORY: Import from P-007A - Decorators and utilities
from src.utils.decorators import retry, time_execution

# validate_signal was removed - using Pydantic validation instead


class AdaptiveMomentumStrategy(BaseStrategy):
    """
    Enhanced Adaptive Momentum Strategy with service layer integration.

    This strategy implements momentum-based trading with dynamic parameter adjustment
    using centralized services for indicators, regime detection, and risk management.
    """

    @property
    def name(self) -> str:
        """Get the strategy name."""
        return getattr(
            self,
            "_name",
            self.config.name if hasattr(self, "config") else "AdaptiveMomentumStrategyRefactored",
        )

    @name.setter
    def name(self, value: str) -> None:
        """Set the strategy name."""
        self._name = value

    @property
    def version(self) -> str:
        """Get the strategy version."""
        return getattr(self, "_version", "2.0.0-refactored")

    @version.setter
    def version(self, value: str) -> None:
        """Set the strategy version."""
        self._version = value

    @property
    def status(self):
        """Get the current strategy status."""
        return getattr(self, "_status", "inactive")

    @status.setter
    def status(self, value) -> None:
        """Set the strategy status."""
        self._status = value

    def __init__(self, config: dict[str, Any]):
        """Initialize adaptive momentum strategy with enhanced architecture."""
        super().__init__(config)

        # Override version for refactored strategy
        self._version = "2.0.0-refactored"

        # Strategy type
        self._strategy_type = StrategyType.MOMENTUM

        # Strategy-specific parameters with defaults from config
        self.fast_ma_period = self.config.parameters.get("fast_ma_period", 20)
        self.slow_ma_period = self.config.parameters.get("slow_ma_period", 50)
        self.rsi_period = self.config.parameters.get("rsi_period", 14)
        self.rsi_overbought = self.config.parameters.get("rsi_overbought", 70)
        self.rsi_oversold = self.config.parameters.get("rsi_oversold", 30)
        self.momentum_lookback = self.config.parameters.get("momentum_lookback", 10)
        self.volume_threshold = self.config.parameters.get("volume_threshold", 1.5)
        self.min_data_points = self.config.parameters.get("min_data_points", 50)

        # Service layer integrations (injected via BaseStrategy)
        self._technical_indicators: TechnicalIndicators | None = None
        self._strategy_service: StrategyService | None = None
        self.regime_detector: MarketRegimeDetector | None = None
        self.adaptive_risk_manager: AdaptiveRiskManager | None = None

        # Strategy state (now managed through service layer)
        self._strategy_state: dict[str, Any] = {
            "momentum_scores": {},
            "last_signals": {},
            "regime_history": {},
            "performance_metrics": {
                "total_momentum_signals": 0,
                "regime_transitions": 0,
                "avg_confidence": 0.0,
            },
        }

        self.logger.info(
            "Enhanced adaptive momentum strategy initialized",
            strategy=self.name,
            version="2.0.0",
            parameters={
                "fast_ma": self.fast_ma_period,
                "slow_ma": self.slow_ma_period,
                "rsi_period": self.rsi_period,
                "momentum_lookback": self.momentum_lookback,
            },
        )

    @property
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        return self._strategy_type

    def set_technical_indicators(self, technical_indicators: TechnicalIndicators) -> None:
        """Set technical indicators service for centralized calculations."""
        self._technical_indicators = technical_indicators
        self.logger.info("Technical indicators service set", strategy=self.name)

    def set_strategy_service(self, strategy_service: StrategyService) -> None:
        """Set strategy service for lifecycle management."""
        self._strategy_service = strategy_service
        self.logger.info("Strategy service set", strategy=self.name)

    def set_regime_detector(self, regime_detector: MarketRegimeDetector) -> None:
        """Set the regime detector for adaptive parameter adjustment."""
        self.regime_detector = regime_detector
        self.logger.info("Regime detector set", strategy=self.name)

    def set_adaptive_risk_manager(self, adaptive_risk_manager: AdaptiveRiskManager) -> None:
        """Set the adaptive risk manager for dynamic risk adjustment."""
        self.adaptive_risk_manager = adaptive_risk_manager
        self.logger.info("Adaptive risk manager set", strategy=self.name)

    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """
        Generate adaptive momentum signals using service layer architecture.

        Enhanced implementation:
        1. Uses TechnicalIndicators service for centralized calculations
        2. Uses regime detection service for market state analysis
        3. Implements proper state persistence
        4. Enhanced error handling and monitoring
        5. Comprehensive metrics tracking

        Args:
            data: Market data for signal generation

        Returns:
            List of trading signals with enhanced confidence scoring
        """
        try:
            # Validate input data
            if not data or not data.symbol:
                self.logger.warning(
                    "Invalid market data received",
                    strategy=self.name,
                    data_available=data is not None,
                )
                return []

            symbol = data.symbol

            # Check if we have sufficient historical data through service layer
            if not await self._validate_data_availability(symbol):
                self.logger.debug(
                    "Insufficient historical data for signal generation",
                    strategy=self.name,
                    symbol=symbol,
                )
                return []

            # Get current market regime through service
            current_regime = await self._get_current_regime_via_service(symbol)

            # Calculate momentum indicators using centralized service
            indicators = await self._calculate_momentum_indicators_via_service(symbol, data)
            if not indicators:
                return []

            # Generate signals based on indicators and regime
            signals = await self._generate_momentum_signals_enhanced(
                data, indicators, current_regime
            )

            # Apply regime-aware adjustments
            signals = await self._apply_enhanced_confidence_adjustments(signals, current_regime)

            # Update strategy state and metrics
            await self._update_strategy_state(symbol, signals, indicators, current_regime)

            self.logger.info(
                "Generated enhanced momentum signals",
                strategy=self.name,
                symbol=symbol,
                signals_count=len(signals),
                regime=current_regime.value if current_regime else "unknown",
                avg_confidence=np.mean([s.confidence for s in signals]) if signals else 0.0,
            )

            return signals

        except Exception as e:
            self.logger.error(
                "Enhanced momentum signal generation failed",
                strategy=self.name,
                symbol=data.symbol if data else "unknown",
                error=str(e),
                exc_info=True,
            )
            return []  # Graceful degradation

    async def _validate_data_availability(self, symbol: str) -> bool:
        """Validate sufficient historical data is available through service layer."""
        try:
            if not self._data_service:
                self.logger.warning("Data service not available", strategy=self.name)
                return False

            # Check data availability through service
            data_count = await self._data_service.get_data_count(symbol)
            return data_count >= self.min_data_points

        except Exception as e:
            self.logger.error(
                "Data availability check failed",
                strategy=self.name,
                symbol=symbol,
                error=str(e),
            )
            return False

    @retry(max_attempts=2, base_delay=1)
    async def _get_current_regime_via_service(self, symbol: str) -> MarketRegime | None:
        """Get current market regime using service layer."""
        try:
            if not self.regime_detector:
                self.logger.debug("Regime detector not available", strategy=self.name)
                return MarketRegime.MEDIUM_VOLATILITY

            # Get recent market data through service
            if not self._data_service:
                return MarketRegime.MEDIUM_VOLATILITY

            recent_data = await self._data_service.get_recent_data(symbol, limit=20)
            if not recent_data:
                return MarketRegime.MEDIUM_VOLATILITY

            # Detect regime using service
            if self.regime_detector:
                regime = await self.regime_detector.detect_comprehensive_regime(recent_data)
            else:
                regime = MarketRegime.MEDIUM_VOLATILITY

            # Update regime history in state
            if symbol not in self._strategy_state["regime_history"]:
                self._strategy_state["regime_history"][symbol] = []

            self._strategy_state["regime_history"][symbol].append(
                {
                    "regime": regime,
                    "timestamp": datetime.now(timezone.utc),
                }
            )

            # Keep only recent history
            if len(self._strategy_state["regime_history"][symbol]) > 50:
                self._strategy_state["regime_history"][symbol] = self._strategy_state[
                    "regime_history"
                ][symbol][-50:]

            return regime

        except Exception as e:
            self.logger.warning(
                "Regime detection via service failed",
                strategy=self.name,
                symbol=symbol,
                error=str(e),
            )
            return MarketRegime.MEDIUM_VOLATILITY

    async def _calculate_momentum_indicators_via_service(
        self, symbol: str, current_data: MarketData
    ) -> dict[str, Any] | None:
        """Calculate momentum indicators using centralized service."""
        try:
            if not self._technical_indicators:
                self.logger.warning(
                    "Technical indicators service not available", strategy=self.name
                )
                return None

            # Get indicators through centralized service
            indicators = {}

            # Moving averages - using the corrected API
            fast_ma = await self._technical_indicators.calculate_sma(symbol, self.fast_ma_period)
            slow_ma = await self._technical_indicators.calculate_sma(symbol, self.slow_ma_period)

            if fast_ma is not None and slow_ma is not None:
                indicators["fast_ma"] = fast_ma
                indicators["slow_ma"] = slow_ma
                indicators["ma_momentum"] = (fast_ma - slow_ma) / slow_ma if slow_ma != 0 else 0.0
            else:
                self.logger.warning(f"Could not calculate moving averages for {symbol}")
                return None

            # RSI calculation - using the corrected API
            rsi = await self._technical_indicators.calculate_rsi(symbol, self.rsi_period)
            if rsi is not None:
                indicators["rsi"] = rsi
                indicators["rsi_score"] = self._calculate_rsi_score_from_value(rsi)
            else:
                indicators["rsi"] = 50.0
                indicators["rsi_score"] = 0.0
                self.logger.debug(f"Could not calculate RSI for {symbol}, using defaults")

            # Momentum calculation - using the corrected API
            price_momentum = await self._technical_indicators.calculate_momentum(
                symbol, self.momentum_lookback
            )
            indicators["price_momentum"] = price_momentum or 0.0

            # Volume analysis - using the corrected API
            volume_ratio = await self._technical_indicators.calculate_volume_ratio(
                symbol, period=20
            )
            indicators["volume_ratio"] = volume_ratio or 1.0
            indicators["volume_score"] = (
                min(volume_ratio / self.volume_threshold, 1.0) if volume_ratio else 0.0
            )

            # Calculate combined momentum score
            momentum_score = (
                indicators["ma_momentum"] * 0.4
                + indicators["price_momentum"] * 0.4
                + indicators["rsi_score"] * 0.2
            )

            # Apply volatility adjustment - using the corrected API
            volatility = await self._technical_indicators.calculate_volatility(symbol, period=20)
            if volatility and volatility > 0:
                volatility_adjustment = 1.0 / (1.0 + volatility)
                momentum_score *= volatility_adjustment

            indicators["combined_momentum_score"] = np.clip(momentum_score, -1.0, 1.0)
            indicators["volatility"] = volatility or 0.0

            self.logger.debug(
                f"Calculated momentum indicators for {symbol}",
                indicators={
                    "fast_ma": indicators["fast_ma"],
                    "slow_ma": indicators["slow_ma"],
                    "rsi": indicators["rsi"],
                    "combined_score": indicators["combined_momentum_score"],
                },
            )

            return indicators

        except Exception as e:
            self.logger.error(
                "Momentum indicators calculation via service failed",
                strategy=self.name,
                symbol=symbol,
                error=str(e),
            )
            return None

    def _calculate_rsi_score_from_value(self, rsi: float) -> float:
        """Convert RSI value to momentum score."""
        if rsi > self.rsi_overbought:
            return -0.5  # Overbought, potential reversal
        elif rsi < self.rsi_oversold:
            return 0.5  # Oversold, potential reversal
        else:
            return (rsi - 50) / 50  # Normal range conversion

    async def _generate_momentum_signals_enhanced(
        self, data: MarketData, indicators: dict[str, Any], current_regime: MarketRegime | None
    ) -> list[Signal]:
        """Generate enhanced momentum signals with comprehensive analysis."""
        signals = []

        try:
            combined_score = indicators.get("combined_momentum_score", 0.0)
            volume_score = indicators.get("volume_score", 0.0)
            volatility = indicators.get("volatility", 0.0)

            # Enhanced signal generation with multiple criteria
            signal_threshold = 0.25

            # Bullish momentum signal
            if combined_score > signal_threshold:
                confidence = self._calculate_enhanced_confidence(
                    combined_score, volume_score, volatility, current_regime, "BUY"
                )

                if confidence >= self.config.min_confidence:
                    signal = Signal(
                        direction=SignalDirection.BUY,
                        confidence=confidence,
                        timestamp=data.timestamp,
                        symbol=data.symbol,
                        strategy_name=self.name,
                        metadata={
                            **indicators,
                            "regime": current_regime.value if current_regime else "unknown",
                            "signal_type": "momentum_bullish",
                            "generation_method": "enhanced_service_layer",
                        },
                    )
                    signals.append(signal)

            # Bearish momentum signal
            elif combined_score < -signal_threshold:
                confidence = self._calculate_enhanced_confidence(
                    combined_score, volume_score, volatility, current_regime, "SELL"
                )

                if confidence >= self.config.min_confidence:
                    signal = Signal(
                        direction=SignalDirection.SELL,
                        confidence=confidence,
                        timestamp=data.timestamp,
                        symbol=data.symbol,
                        strategy_name=self.name,
                        metadata={
                            **indicators,
                            "regime": current_regime.value if current_regime else "unknown",
                            "signal_type": "momentum_bearish",
                            "generation_method": "enhanced_service_layer",
                        },
                    )
                    signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(
                "Enhanced momentum signal generation failed",
                strategy=self.name,
                symbol=data.symbol,
                error=str(e),
            )
            return []

    def _calculate_enhanced_confidence(
        self,
        momentum_score: float,
        volume_score: float,
        volatility: float,
        regime: MarketRegime | None,
        direction: str,
    ) -> float:
        """Calculate enhanced confidence score with multiple factors."""
        try:
            # Base confidence from momentum strength
            base_confidence = min(abs(momentum_score), 0.9)

            # Volume confirmation boost
            volume_boost = volume_score * 0.1

            # Volatility adjustment (lower volatility = higher confidence)
            volatility_adjustment = 1.0 - min(volatility * 2, 0.3)

            # Regime-specific adjustments
            regime_multiplier = self._get_regime_confidence_multiplier(regime)

            # Combine all factors
            enhanced_confidence = (
                (base_confidence + volume_boost) * volatility_adjustment * regime_multiplier
            )

            return min(enhanced_confidence, 0.95)

        except Exception as e:
            self.logger.error("Enhanced confidence calculation failed", error=str(e))
            return 0.1

    async def _apply_enhanced_confidence_adjustments(
        self, signals: list[Signal], current_regime: MarketRegime | None
    ) -> list[Signal]:
        """Apply enhanced confidence adjustments using service layer."""
        try:
            if not signals or not current_regime:
                return signals

            adjusted_signals = []
            for signal in signals:
                # Get adaptive parameters if available
                adaptive_params = None
                if self.adaptive_risk_manager:
                    adaptive_params = self.adaptive_risk_manager.get_adaptive_parameters(
                        current_regime
                    )

                # Apply regime-specific adjustments
                regime_multiplier = self._get_regime_confidence_multiplier(current_regime)
                adjusted_confidence = signal.confidence * regime_multiplier

                # Update signal with enhanced metadata
                signal.confidence = min(adjusted_confidence, 0.95)
                signal.metadata.update(
                    {
                        "regime_confidence_multiplier": regime_multiplier,
                        "adaptive_params": adaptive_params,
                        "enhancement_version": "2.0.0",
                    }
                )

                adjusted_signals.append(signal)

            self.logger.debug(
                "Applied enhanced confidence adjustments",
                strategy=self.name,
                original_count=len(signals),
                adjusted_count=len(adjusted_signals),
                regime=current_regime.value,
            )

            return adjusted_signals

        except Exception as e:
            self.logger.error(
                "Enhanced confidence adjustment failed",
                strategy=self.name,
                error=str(e),
            )
            return signals

    async def _update_strategy_state(
        self,
        symbol: str,
        signals: list[Signal],
        indicators: dict[str, Any],
        current_regime: MarketRegime | None,
    ) -> None:
        """Update strategy state with current analysis results."""
        try:
            # Update momentum scores
            self._strategy_state["momentum_scores"][symbol] = {
                "score": indicators.get("combined_momentum_score", 0.0),
                "timestamp": datetime.now(timezone.utc),
                "components": {
                    "ma_momentum": indicators.get("ma_momentum", 0.0),
                    "price_momentum": indicators.get("price_momentum", 0.0),
                    "rsi_score": indicators.get("rsi_score", 0.0),
                },
            }

            # Update last signals
            if signals:
                self._strategy_state["last_signals"][symbol] = {
                    "signals": [s.model_dump() for s in signals],
                    "timestamp": datetime.now(timezone.utc),
                    "count": len(signals),
                }

            # Update performance metrics
            self._strategy_state["performance_metrics"]["total_momentum_signals"] += len(signals)

            if signals:
                current_avg = self._strategy_state["performance_metrics"]["avg_confidence"]
                signal_avg = np.mean([s.confidence for s in signals])
                total_signals = self._strategy_state["performance_metrics"][
                    "total_momentum_signals"
                ]

                # Update rolling average confidence
                if total_signals > 1:
                    new_avg = (
                        (current_avg * (total_signals - len(signals))) + (signal_avg * len(signals))
                    ) / total_signals
                else:
                    new_avg = signal_avg

                self._strategy_state["performance_metrics"]["avg_confidence"] = new_avg

            # Persist state through service if available
            if self._strategy_service:
                await self._persist_strategy_state()

        except Exception as e:
            self.logger.error(
                "Strategy state update failed",
                strategy=self.name,
                symbol=symbol,
                error=str(e),
            )

    async def _persist_strategy_state(self) -> None:
        """Persist strategy state through service layer."""
        try:
            if self._strategy_service:
                await self._strategy_service.persist_strategy_state(
                    self.config.strategy_id, self._strategy_state
                )
        except Exception as e:
            self.logger.error(
                "Strategy state persistence failed",
                strategy=self.name,
                error=str(e),
            )

    def _get_regime_confidence_multiplier(self, regime: MarketRegime | None) -> float:
        """Get enhanced regime-specific confidence multiplier."""
        if not regime:
            return 1.0

        confidence_multipliers = {
            MarketRegime.LOW_VOLATILITY: 1.15,  # Higher confidence in low vol
            MarketRegime.MEDIUM_VOLATILITY: 1.0,  # Standard confidence
            MarketRegime.HIGH_VOLATILITY: 0.75,  # Lower confidence in high vol
            MarketRegime.CRISIS: 0.5,  # Much lower confidence in crisis
            MarketRegime.TRENDING_UP: 1.1,  # Higher in uptrend (momentum favors trends)
            MarketRegime.TRENDING_DOWN: 0.85,  # Slightly lower in downtrend
            MarketRegime.RANGING: 0.9,  # Lower in ranging (momentum less effective)
            MarketRegime.HIGH_CORRELATION: 0.85,  # Lower in high correlation
            MarketRegime.LOW_CORRELATION: 1.1,  # Higher in low correlation
        }

        return confidence_multipliers.get(regime, 1.0)

    @retry(max_attempts=2, base_delay=1)
    async def validate_signal(self, signal: Signal) -> bool:
        """Enhanced signal validation with service layer integration."""
        try:
            # Basic signal validation
            if not signal or not signal.symbol or not signal.direction:
                return False

            # Enhanced strategy-specific validation
            symbol = signal.symbol

            # Check if we have recent momentum data in state
            momentum_data = self._strategy_state["momentum_scores"].get(symbol)
            if not momentum_data:
                self.logger.warning(
                    "No recent momentum data for signal validation",
                    strategy=self.name,
                    symbol=symbol,
                )
                return False

            # Validate momentum score consistency with tolerance
            signal_momentum = signal.metadata.get("combined_momentum_score", 0.0)
            current_momentum = momentum_data.get("score", 0.0)

            tolerance = 0.25  # Increased tolerance for enhanced algorithm
            if abs(current_momentum - signal_momentum) > tolerance:
                self.logger.warning(
                    "Momentum score inconsistency detected",
                    strategy=self.name,
                    symbol=symbol,
                    current_momentum=current_momentum,
                    signal_momentum=signal_momentum,
                    tolerance=tolerance,
                )
                return False

            # Validate signal freshness (within last 5 minutes)
            signal_age = (datetime.now(timezone.utc) - signal.timestamp).total_seconds()
            if signal_age > 300:  # 5 minutes
                self.logger.warning(
                    "Signal too old for validation",
                    strategy=self.name,
                    symbol=symbol,
                    age_seconds=signal_age,
                )
                return False

            # Validate confidence is above minimum threshold
            if signal.confidence < self.config.min_confidence:
                self.logger.warning(
                    "Signal confidence below threshold",
                    strategy=self.name,
                    symbol=symbol,
                    confidence=signal.confidence,
                    min_threshold=self.config.min_confidence,
                )
                return False

            self.logger.debug(
                "Enhanced momentum signal validated",
                strategy=self.name,
                symbol=symbol,
                direction=signal.direction.value,
                confidence=signal.confidence,
                validation_version="2.0.0",
            )

            return True

        except Exception as e:
            self.logger.error(
                "Enhanced signal validation failed",
                strategy=self.name,
                symbol=signal.symbol if signal else "unknown",
                error=str(e),
            )
            return False

    def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size using enhanced risk management."""
        try:
            # Use adaptive risk manager if available
            if self.adaptive_risk_manager:
                # Get current regime from signal metadata
                regime_str = signal.metadata.get("regime", "medium_volatility")
                try:
                    current_regime = MarketRegime(regime_str)
                except ValueError:
                    current_regime = MarketRegime.MEDIUM_VOLATILITY

                # Get portfolio value from risk manager or service
                portfolio_value = Decimal("10000")  # TODO: Get from portfolio service
                if self._risk_manager:
                    try:
                        portfolio_value = self._risk_manager.get_available_capital()
                    except Exception:
                        pass

                # Calculate position size with enhanced factors
                volatility = signal.metadata.get("volatility", 0.02)
                volume_score = signal.metadata.get("volume_score", 1.0)

                # Adjust position size based on volatility and volume
                base_position_pct = self.config.position_size_pct
                volatility_adjustment = max(
                    0.5, 1.0 - (volatility * 10)
                )  # Reduce size in high volatility
                volume_adjustment = min(1.5, volume_score)  # Increase size with high volume

                adjusted_position_pct = (
                    base_position_pct * volatility_adjustment * volume_adjustment
                )
                position_size = portfolio_value * Decimal(str(adjusted_position_pct))

                self.logger.info(
                    "Enhanced position sizing calculated",
                    strategy=self.name,
                    symbol=signal.symbol,
                    position_size=float(position_size),
                    regime=current_regime.value,
                    adjustments={
                        "volatility": volatility_adjustment,
                        "volume": volume_adjustment,
                    },
                )

                return position_size
            else:
                # Enhanced fallback calculation
                base_size = Decimal(str(self.config.position_size_pct))
                confidence_adjustment = Decimal(str(signal.confidence))
                adjusted_size = base_size * confidence_adjustment

                self.logger.info(
                    "Enhanced fallback position sizing",
                    strategy=self.name,
                    symbol=signal.symbol,
                    position_size=float(adjusted_size),
                    confidence_adjustment=float(confidence_adjustment),
                )
                return adjusted_size

        except Exception as e:
            self.logger.error(
                "Enhanced position size calculation failed",
                strategy=self.name,
                symbol=signal.symbol,
                error=str(e),
            )
            return Decimal(str(self.config.position_size_pct))

    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Enhanced exit logic with service layer integration."""
        try:
            symbol = position.symbol

            # Check current momentum from strategy state
            momentum_data = self._strategy_state["momentum_scores"].get(symbol)
            if not momentum_data:
                self.logger.warning(
                    "No momentum data for exit decision",
                    strategy=self.name,
                    symbol=symbol,
                )
                return False

            current_momentum = momentum_data.get("score", 0.0)
            entry_momentum = position.metadata.get("entry_momentum", 0.0)

            # Enhanced momentum reversal logic
            reversal_threshold = 0.3  # More sensitive threshold
            position_side = position.side.value.lower()

            momentum_reversal = False
            if position_side == "buy" and current_momentum < -reversal_threshold:
                momentum_reversal = True
            elif position_side == "sell" and current_momentum > reversal_threshold:
                momentum_reversal = True

            if momentum_reversal:
                self.logger.info(
                    "Enhanced momentum reversal exit triggered",
                    strategy=self.name,
                    symbol=symbol,
                    position_side=position_side,
                    entry_momentum=entry_momentum,
                    current_momentum=current_momentum,
                    threshold=reversal_threshold,
                )
                return True

            # Additional exit conditions based on regime changes
            if self.regime_detector:
                try:
                    current_regime_data = self._strategy_state["regime_history"].get(symbol, [])
                    if current_regime_data:
                        recent_regime = current_regime_data[-1]["regime"]
                        entry_regime = position.metadata.get("entry_regime")

                        # Exit if regime has changed unfavorably
                        if (
                            entry_regime
                            and recent_regime != entry_regime
                            and recent_regime in [MarketRegime.CRISIS, MarketRegime.HIGH_VOLATILITY]
                        ):
                            self.logger.info(
                                "Regime change exit triggered",
                                strategy=self.name,
                                symbol=symbol,
                                entry_regime=entry_regime,
                                current_regime=recent_regime.value,
                            )
                            return True
                except Exception as regime_error:
                    self.logger.debug(
                        "Regime exit check failed",
                        strategy=self.name,
                        error=str(regime_error),
                    )

            return False

        except Exception as e:
            self.logger.error(
                "Enhanced exit condition check failed",
                strategy=self.name,
                symbol=position.symbol if position else "unknown",
                error=str(e),
            )
            return False

    def get_strategy_info(self) -> dict[str, Any]:
        """Get enhanced strategy information with service layer integration."""
        base_info = super().get_strategy_info()

        # Add enhanced integration information
        enhanced_info = {
            "version": "2.0.0",
            "architecture": "service_layer_enhanced",
            "strategy_state": {
                "momentum_scores_count": len(self._strategy_state["momentum_scores"]),
                "last_signals_count": len(self._strategy_state["last_signals"]),
                "regime_history_count": len(self._strategy_state["regime_history"]),
                "performance_metrics": self._strategy_state["performance_metrics"],
            },
            "service_integrations": {
                "technical_indicators_available": self._technical_indicators is not None,
                "strategy_service_available": self._strategy_service is not None,
                "regime_detector_available": self.regime_detector is not None,
                "adaptive_risk_manager_available": self.adaptive_risk_manager is not None,
                "data_service_available": self._data_service is not None,
                "risk_manager_available": self._risk_manager is not None,
            },
            "configuration": {
                "fast_ma_period": self.fast_ma_period,
                "slow_ma_period": self.slow_ma_period,
                "rsi_period": self.rsi_period,
                "momentum_lookback": self.momentum_lookback,
                "volume_threshold": self.volume_threshold,
                "min_data_points": self.min_data_points,
            },
            "refactoring_status": {
                "centralized_indicators": True,
                "service_layer_integration": True,
                "state_persistence": True,
                "enhanced_metrics": True,
                "removed_direct_access": True,
            },
        }

        base_info.update(enhanced_info)
        return base_info

    async def _on_start(self) -> None:
        """Enhanced strategy startup with service validations."""
        # Validate required services are available
        if not self._technical_indicators:
            self.logger.warning(
                "Technical indicators service not available - "
                "strategy may have limited functionality",
                strategy=self.name,
            )

        if not self._data_service:
            self.logger.warning(
                "Data service not available - strategy may have limited functionality",
                strategy=self.name,
            )

        self.logger.info(
            "Enhanced adaptive momentum strategy started",
            strategy=self.name,
            version="2.0.0",
        )

    async def _on_stop(self) -> None:
        """Enhanced strategy shutdown with state persistence."""
        try:
            # Persist final state
            await self._persist_strategy_state()

            self.logger.info(
                "Enhanced adaptive momentum strategy stopped",
                strategy=self.name,
                final_metrics=self._strategy_state["performance_metrics"],
            )
        except Exception as e:
            self.logger.error(
                "Error during enhanced strategy shutdown",
                strategy=self.name,
                error=str(e),
            )

    def cleanup(self) -> None:
        """Enhanced cleanup with state management."""
        try:
            # Clear strategy state
            self._strategy_state = {
                "momentum_scores": {},
                "last_signals": {},
                "regime_history": {},
                "performance_metrics": {
                    "total_momentum_signals": 0,
                    "regime_transitions": 0,
                    "avg_confidence": 0.0,
                },
            }

            self.logger.info(
                "Enhanced adaptive momentum strategy cleanup completed",
                strategy=self.name,
            )

        except Exception as e:
            self.logger.error(
                "Error during enhanced cleanup",
                strategy=self.name,
                error=str(e),
            )
        finally:
            super().cleanup()
