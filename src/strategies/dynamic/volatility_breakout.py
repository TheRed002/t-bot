"""
Volatility Breakout Strategy Implementation - REFACTORED Day 13

This module implements a volatility-based breakout strategy using the enhanced
BaseStrategy architecture with service layer integration, centralized indicator
calculations, and state persistence.

Refactoring improvements:
- Uses StrategyService for lifecycle management
- Uses TechnicalIndicators service for centralized ATR and volatility calculations
- Implements proper state persistence
- Enhanced metrics tracking and monitoring
- Removed direct data access patterns
- Advanced consolidation detection through services

CRITICAL: This strategy follows the perfect architecture from Day 12.
"""

from datetime import datetime, timedelta
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


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Enhanced Volatility Breakout Strategy with service layer integration.

    This strategy implements breakout trading using ATR-based thresholds and
    integrates with centralized services for indicators, regime detection, and risk management.
    """

    @property
    def name(self) -> str:
        """Get the strategy name."""
        return getattr(
            self,
            "_name",
            self.config.name if hasattr(self, "config") else "VolatilityBreakoutStrategyRefactored",
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
        """Initialize volatility breakout strategy with enhanced architecture."""
        super().__init__(config)

        # Override version for refactored strategy
        self._version = "2.0.0-refactored"

        # Strategy type
        self._strategy_type = StrategyType.TREND_FOLLOWING

        # Strategy-specific parameters with defaults from config
        self.atr_period = self.config.parameters.get("atr_period", 14)
        self.breakout_multiplier = self.config.parameters.get("breakout_multiplier", 2.0)
        self.consolidation_period = self.config.parameters.get("consolidation_period", 20)
        self.volume_confirmation = self.config.parameters.get("volume_confirmation", True)
        self.min_consolidation_ratio = self.config.parameters.get("min_consolidation_ratio", 0.8)
        self.max_consolidation_ratio = self.config.parameters.get("max_consolidation_ratio", 1.2)
        self.time_decay_factor = self.config.parameters.get("time_decay_factor", 0.95)
        self.min_data_points = self.config.parameters.get("min_data_points", 50)
        self.breakout_cooldown_minutes = self.config.parameters.get("breakout_cooldown_minutes", 30)

        # Service layer integrations (injected via BaseStrategy)
        self._technical_indicators: TechnicalIndicators | None = None
        self._strategy_service: StrategyService | None = None
        self.regime_detector: MarketRegimeDetector | None = None
        self.adaptive_risk_manager: AdaptiveRiskManager | None = None

        # Strategy state (now managed through service layer)
        self._strategy_state: dict[str, Any] = {
            "atr_values": {},
            "breakout_levels": {},
            "consolidation_scores": {},
            "last_breakout_times": {},
            "regime_history": {},
            "performance_metrics": {
                "total_breakout_signals": 0,
                "successful_breakouts": 0,
                "false_breakouts": 0,
                "avg_consolidation_score": 0.0,
                "regime_transitions": 0,
            },
        }

        self.logger.info(
            "Enhanced volatility breakout strategy initialized",
            strategy=self.name,
            version="2.0.0",
            parameters={
                "atr_period": self.atr_period,
                "breakout_multiplier": self.breakout_multiplier,
                "consolidation_period": self.consolidation_period,
                "time_decay_factor": self.time_decay_factor,
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
        Generate volatility breakout signals using service layer architecture.

        Enhanced implementation:
        1. Uses TechnicalIndicators service for ATR and volatility calculations
        2. Uses regime detection service for market state analysis
        3. Implements consolidation detection through centralized services
        4. Enhanced breakout validation and time-decay logic
        5. Comprehensive metrics tracking

        Args:
            data: Market data for signal generation

        Returns:
            List of trading signals with enhanced breakout analysis
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

            # Check breakout cooldown period
            if await self._is_in_breakout_cooldown(symbol):
                self.logger.debug(
                    "Still in breakout cooldown period",
                    strategy=self.name,
                    symbol=symbol,
                )
                return []

            # Get current market regime through service
            current_regime = await self._get_current_regime_via_service(symbol)

            # Calculate volatility indicators using centralized service
            indicators = await self._calculate_volatility_indicators_via_service(symbol, data)
            if not indicators:
                return []

            # Calculate breakout levels with regime adjustments
            breakout_levels = await self._calculate_breakout_levels_enhanced(
                symbol, indicators, current_regime
            )

            # Generate breakout signals
            signals = await self._generate_breakout_signals_enhanced(
                data, indicators, breakout_levels, current_regime
            )

            # Apply regime-specific filtering
            signals = await self._apply_enhanced_regime_filtering(signals, current_regime)

            # Apply time-decay adjustments
            signals = await self._apply_enhanced_time_decay(signals, symbol)

            # Update strategy state and metrics
            await self._update_strategy_state(symbol, signals, indicators, current_regime)

            self.logger.info(
                "Generated enhanced volatility breakout signals",
                strategy=self.name,
                symbol=symbol,
                signals_count=len(signals),
                regime=current_regime.value if current_regime else "unknown",
                atr_value=indicators.get("atr", 0.0),
                consolidation_score=indicators.get("consolidation_score", 0.0),
            )

            return signals

        except Exception as e:
            self.logger.error(
                "Enhanced breakout signal generation failed",
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

    async def _is_in_breakout_cooldown(self, symbol: str) -> bool:
        """Check if symbol is still in breakout cooldown period."""
        try:
            last_breakout_times = self._strategy_state["last_breakout_times"]
            if symbol not in last_breakout_times:
                return False

            last_breakout = last_breakout_times[symbol]
            cooldown_period = timedelta(minutes=self.breakout_cooldown_minutes)

            return (datetime.now() - last_breakout) < cooldown_period

        except Exception as e:
            self.logger.error(
                "Breakout cooldown check failed",
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
            regime = await self.regime_detector.detect_comprehensive_regime(recent_data)

            # Update regime history in state
            if symbol not in self._strategy_state["regime_history"]:
                self._strategy_state["regime_history"][symbol] = []

            self._strategy_state["regime_history"][symbol].append(
                {
                    "regime": regime,
                    "timestamp": datetime.now(),
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

    async def _calculate_volatility_indicators_via_service(
        self, symbol: str, current_data: MarketData
    ) -> dict[str, Any] | None:
        """Calculate volatility indicators using centralized service."""
        try:
            if not self._technical_indicators:
                self.logger.warning(
                    "Technical indicators service not available", strategy=self.name
                )
                return None

            # Get indicators through centralized service
            indicators = {}

            # ATR calculation - using the corrected API
            atr = await self._technical_indicators.calculate_atr(symbol, self.atr_period)
            if atr is None:
                self.logger.warning(f"Could not calculate ATR for {symbol}")
                return None
            indicators["atr"] = atr

            # Historical volatility - using the corrected API
            volatility = await self._technical_indicators.calculate_volatility(
                symbol, period=self.consolidation_period
            )
            indicators["historical_volatility"] = volatility or 0.0

            # Price range analysis for consolidation detection
            if not self._data_service:
                self.logger.warning("Data service not available for price range analysis")
                return None

            price_data = await self._data_service.get_recent_data(
                symbol, limit=self.consolidation_period
            )
            if not price_data:
                self.logger.warning(
                    f"No price data available for consolidation analysis of {symbol}"
                )
                return None

            # Calculate consolidation score using service data
            consolidation_score = await self._calculate_consolidation_score_enhanced(
                symbol, price_data
            )
            indicators["consolidation_score"] = consolidation_score

            # Volume analysis - using the corrected API
            volume_ratio = await self._technical_indicators.calculate_volume_ratio(
                symbol, period=20
            )
            indicators["volume_ratio"] = volume_ratio or 1.0
            indicators["volume_confirmation"] = volume_ratio > 1.2 if volume_ratio else False

            # Bollinger Band analysis for breakout context - using the corrected API
            bb_data = await self._technical_indicators.calculate_bollinger_bands(
                symbol, period=20, std_dev=2.0
            )
            if bb_data:
                indicators["bb_squeeze"] = self._is_bollinger_squeeze(bb_data, current_data.price)
                indicators["bb_position"] = self._get_bb_position(bb_data, current_data.price)
            else:
                indicators["bb_squeeze"] = False
                indicators["bb_position"] = "unknown"
                self.logger.debug(f"Could not calculate Bollinger Bands for {symbol}")

            self.logger.debug(
                f"Calculated volatility indicators for {symbol}",
                indicators={
                    "atr": indicators["atr"],
                    "volatility": indicators["historical_volatility"],
                    "consolidation_score": indicators["consolidation_score"],
                    "volume_ratio": indicators["volume_ratio"],
                },
            )

            return indicators

        except Exception as e:
            self.logger.error(
                "Volatility indicators calculation via service failed",
                strategy=self.name,
                symbol=symbol,
                error=str(e),
            )
            return None

    async def _calculate_consolidation_score_enhanced(
        self, symbol: str, price_data: list[MarketData]
    ) -> float:
        """Calculate enhanced consolidation score using service data."""
        try:
            if len(price_data) < self.consolidation_period:
                return 0.0

            prices = [float(data.price) for data in price_data]

            # Calculate price range during consolidation period
            price_array = np.array(prices)
            price_range = np.max(price_array) - np.min(price_array)
            avg_price = np.mean(price_array)

            # Consolidation ratio (range relative to average price)
            consolidation_ratio = price_range / avg_price if avg_price > 0 else 0.0

            # Calculate price stability (lower standard deviation = higher consolidation)
            price_stability = 1.0 / (1.0 + np.std(price_array) / avg_price)

            # Volume consistency analysis if available
            volume_consistency = 1.0  # Default
            if all(hasattr(data, "volume") and data.volume for data in price_data):
                volumes = [float(data.volume) for data in price_data]
                volume_std = np.std(volumes)
                volume_mean = np.mean(volumes)
                volume_consistency = (
                    1.0 / (1.0 + volume_std / volume_mean) if volume_mean > 0 else 1.0
                )

            # Enhanced consolidation score calculation
            if self.min_consolidation_ratio <= consolidation_ratio <= self.max_consolidation_ratio:
                # Good consolidation - narrow range
                base_score = 1.0 - (consolidation_ratio - self.min_consolidation_ratio) / (
                    self.max_consolidation_ratio - self.min_consolidation_ratio
                )

                # Apply stability and volume consistency adjustments
                enhanced_score = base_score * price_stability * volume_consistency

                # Additional penalty for too recent volatility
                recent_volatility = (
                    np.std(price_array[-5:]) / np.mean(price_array[-5:])
                    if len(price_array) >= 5
                    else 0
                )
                volatility_penalty = max(0.5, 1.0 - recent_volatility * 10)

                enhanced_score *= volatility_penalty
            else:
                # Poor consolidation - too wide or too narrow
                enhanced_score = 0.0

            return min(max(enhanced_score, 0.0), 1.0)

        except Exception as e:
            self.logger.error(
                "Enhanced consolidation score calculation failed",
                strategy=self.name,
                symbol=symbol,
                error=str(e),
            )
            return 0.0

    def _is_bollinger_squeeze(self, bb_data: dict, current_price: Decimal) -> bool:
        """Detect if market is in Bollinger Band squeeze (consolidation)."""
        try:
            if not bb_data or "upper" not in bb_data or "lower" not in bb_data:
                return False

            band_width = float(bb_data["upper"] - bb_data["lower"])
            middle_price = float(bb_data.get("middle", current_price))

            # Consider it a squeeze if band width is less than 4% of middle price
            return band_width / middle_price < 0.04 if middle_price > 0 else False

        except Exception:
            return False

    def _get_bb_position(self, bb_data: dict, current_price: Decimal) -> str:
        """Get current price position relative to Bollinger Bands."""
        try:
            if not bb_data or "upper" not in bb_data or "lower" not in bb_data:
                return "unknown"

            price = float(current_price)
            upper = float(bb_data["upper"])
            lower = float(bb_data["lower"])

            if price > upper:
                return "above_upper"
            elif price < lower:
                return "below_lower"
            else:
                return "inside_bands"

        except Exception:
            return "unknown"

    async def _calculate_breakout_levels_enhanced(
        self, symbol: str, indicators: dict[str, Any], current_regime: MarketRegime | None
    ) -> dict[str, float]:
        """Calculate enhanced breakout levels using ATR and regime adjustments."""
        try:
            if not self._data_service:
                return {}

            # Get current price
            recent_data = await self._data_service.get_recent_data(symbol, limit=1)
            if not recent_data:
                return {}

            current_price = float(recent_data[0].price)
            atr_value = indicators.get("atr", 0.0)

            # Calculate base breakout distance
            base_distance = atr_value * self.breakout_multiplier

            # Apply regime-specific adjustments
            regime_adjustment = self._get_regime_breakout_adjustment(current_regime)

            # Apply consolidation quality adjustment
            consolidation_score = indicators.get("consolidation_score", 0.0)
            consolidation_adjustment = 0.8 + (consolidation_score * 0.4)  # 0.8 to 1.2 range

            # Apply volatility adjustment
            volatility = indicators.get("historical_volatility", 0.02)
            volatility_adjustment = max(0.7, min(1.3, 1.0 + (volatility - 0.02) * 10))

            adjusted_distance = (
                base_distance * regime_adjustment * consolidation_adjustment * volatility_adjustment
            )

            breakout_levels = {
                "upper_breakout": current_price + adjusted_distance,
                "lower_breakout": current_price - adjusted_distance,
                "atr_value": atr_value,
                "base_distance": base_distance,
                "adjusted_distance": adjusted_distance,
                "regime_adjustment": regime_adjustment,
                "consolidation_adjustment": consolidation_adjustment,
                "volatility_adjustment": volatility_adjustment,
                "current_price": current_price,
            }

            # Store in strategy state
            self._strategy_state["breakout_levels"][symbol] = breakout_levels
            self._strategy_state["atr_values"][symbol] = atr_value

            return breakout_levels

        except Exception as e:
            self.logger.error(
                "Enhanced breakout levels calculation failed",
                strategy=self.name,
                symbol=symbol,
                error=str(e),
            )
            return {}

    def _get_regime_breakout_adjustment(self, regime: MarketRegime | None) -> float:
        """Get enhanced regime-specific breakout threshold adjustment."""
        if not regime:
            return 1.0

        regime_adjustments = {
            MarketRegime.LOW_VOLATILITY: 1.4,  # Wider breakout levels in low vol
            MarketRegime.MEDIUM_VOLATILITY: 1.0,  # Standard breakout levels
            MarketRegime.HIGH_VOLATILITY: 0.7,  # Tighter breakout levels in high vol
            MarketRegime.CRISIS: 0.5,  # Very tight levels in crisis
            MarketRegime.TRENDING_UP: 1.1,  # Slightly wider levels in uptrend
            MarketRegime.TRENDING_DOWN: 0.9,  # Slightly tighter levels in downtrend
            MarketRegime.RANGING: 1.2,  # Wider in ranging (need real breakouts)
            MarketRegime.HIGH_CORRELATION: 0.8,  # Tighter in high correlation
            MarketRegime.LOW_CORRELATION: 1.2,  # Wider in low correlation
        }

        return regime_adjustments.get(regime, 1.0)

    async def _generate_breakout_signals_enhanced(
        self,
        data: MarketData,
        indicators: dict[str, Any],
        breakout_levels: dict[str, float],
        current_regime: MarketRegime | None,
    ) -> list[Signal]:
        """Generate enhanced breakout signals with comprehensive validation."""
        signals = []

        try:
            if not breakout_levels:
                return signals

            current_price = float(data.price)
            consolidation_score = indicators.get("consolidation_score", 0.0)
            volume_confirmation = indicators.get("volume_confirmation", False)
            bb_squeeze = indicators.get("bb_squeeze", False)

            # Enhanced breakout conditions
            min_consolidation = 0.4  # Lower threshold for enhanced algorithm
            volume_required = self.volume_confirmation

            # Upper breakout (bullish)
            if (
                current_price > breakout_levels.get("upper_breakout", float("inf"))
                and consolidation_score > min_consolidation
                and (not volume_required or volume_confirmation)
            ):
                confidence = self._calculate_enhanced_breakout_confidence(
                    "upper", current_price, breakout_levels, indicators, current_regime
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
                            **breakout_levels,
                            "regime": current_regime.value if current_regime else "unknown",
                            "breakout_type": "upper",
                            "signal_type": "volatility_breakout_bullish",
                            "generation_method": "enhanced_service_layer",
                            "bb_squeeze": bb_squeeze,
                        },
                    )
                    signals.append(signal)

            # Lower breakout (bearish)
            elif (
                current_price < breakout_levels.get("lower_breakout", float("-inf"))
                and consolidation_score > min_consolidation
                and (not volume_required or volume_confirmation)
            ):
                confidence = self._calculate_enhanced_breakout_confidence(
                    "lower", current_price, breakout_levels, indicators, current_regime
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
                            **breakout_levels,
                            "regime": current_regime.value if current_regime else "unknown",
                            "breakout_type": "lower",
                            "signal_type": "volatility_breakout_bearish",
                            "generation_method": "enhanced_service_layer",
                            "bb_squeeze": bb_squeeze,
                        },
                    )
                    signals.append(signal)

            # Update last breakout time if signals generated
            if signals:
                self._strategy_state["last_breakout_times"][data.symbol] = data.timestamp

            return signals

        except Exception as e:
            self.logger.error(
                "Enhanced breakout signal generation failed",
                strategy=self.name,
                symbol=data.symbol,
                error=str(e),
            )
            return []

    def _calculate_enhanced_breakout_confidence(
        self,
        breakout_type: str,
        current_price: float,
        breakout_levels: dict[str, float],
        indicators: dict[str, Any],
        regime: MarketRegime | None,
    ) -> float:
        """Calculate enhanced confidence for breakout signals."""
        try:
            # Base confidence from breakout strength
            if breakout_type == "upper":
                breakout_level = breakout_levels.get("upper_breakout", 0.0)
                breakout_strength = (
                    (current_price - breakout_level) / breakout_level if breakout_level > 0 else 0.0
                )
            else:
                breakout_level = breakout_levels.get("lower_breakout", 0.0)
                breakout_strength = (
                    (breakout_level - current_price) / breakout_level if breakout_level > 0 else 0.0
                )

            base_confidence = min(abs(breakout_strength) * 5, 0.8)  # Scale breakout strength

            # Consolidation quality boost
            consolidation_score = indicators.get("consolidation_score", 0.0)
            consolidation_boost = consolidation_score * 0.15

            # Volume confirmation boost
            volume_boost = 0.1 if indicators.get("volume_confirmation", False) else 0.0

            # Bollinger Band squeeze boost (indicates strong consolidation)
            bb_squeeze_boost = 0.05 if indicators.get("bb_squeeze", False) else 0.0

            # ATR consistency check (higher ATR = more significant breakout)
            atr_value = indicators.get("atr", 0.0)
            atr_boost = min(atr_value * 100, 0.1)  # Scale ATR to reasonable boost

            # Regime-specific adjustments
            regime_multiplier = self._get_regime_confidence_multiplier(regime)

            # Combine all factors
            enhanced_confidence = (
                base_confidence + consolidation_boost + volume_boost + bb_squeeze_boost + atr_boost
            ) * regime_multiplier

            return min(enhanced_confidence, 0.95)

        except Exception as e:
            self.logger.error("Enhanced breakout confidence calculation failed", error=str(e))
            return 0.1

    def _get_regime_confidence_multiplier(self, regime: MarketRegime | None) -> float:
        """Get regime-specific confidence multiplier for breakout strategies."""
        if not regime:
            return 1.0

        confidence_multipliers = {
            MarketRegime.LOW_VOLATILITY: 1.2,  # Breakouts more significant in low vol
            MarketRegime.MEDIUM_VOLATILITY: 1.0,  # Standard confidence
            MarketRegime.HIGH_VOLATILITY: 0.7,  # More noise in high vol
            MarketRegime.CRISIS: 0.5,  # Very unreliable in crisis
            MarketRegime.TRENDING_UP: 1.1,  # Slightly higher for trend continuation
            MarketRegime.TRENDING_DOWN: 0.9,  # Slightly lower in downtrend
            MarketRegime.RANGING: 1.15,  # Breakouts from ranges are valuable
            MarketRegime.HIGH_CORRELATION: 0.8,  # Less reliable in correlated markets
            MarketRegime.LOW_CORRELATION: 1.1,  # More reliable when markets diverge
        }

        return confidence_multipliers.get(regime, 1.0)

    async def _apply_enhanced_regime_filtering(
        self, signals: list[Signal], current_regime: MarketRegime | None
    ) -> list[Signal]:
        """Apply enhanced regime-specific filtering for breakout signals."""
        try:
            if not signals or not current_regime:
                return signals

            filtered_signals = []

            for signal in signals:
                if self._is_signal_valid_for_regime_enhanced(signal, current_regime):
                    # Get adaptive parameters if available
                    adaptive_params = None
                    if self.adaptive_risk_manager:
                        adaptive_params = self.adaptive_risk_manager.get_adaptive_parameters(
                            current_regime
                        )

                    # Apply regime-specific confidence adjustments
                    regime_multiplier = self._get_regime_confidence_multiplier(current_regime)
                    signal.confidence = min(signal.confidence * regime_multiplier, 0.95)

                    # Update signal metadata
                    signal.metadata.update(
                        {
                            "regime_confidence_multiplier": regime_multiplier,
                            "adaptive_params": adaptive_params,
                            "regime_filtering_version": "2.0.0",
                        }
                    )

                    filtered_signals.append(signal)
                else:
                    self.logger.debug(
                        "Signal filtered out by enhanced regime conditions",
                        strategy=self.name,
                        symbol=signal.symbol,
                        regime=current_regime.value,
                        signal_confidence=signal.confidence,
                    )

            self.logger.debug(
                "Applied enhanced regime filtering",
                strategy=self.name,
                original_count=len(signals),
                filtered_count=len(filtered_signals),
                regime=current_regime.value,
            )

            return filtered_signals

        except Exception as e:
            self.logger.error(
                "Enhanced regime filtering failed",
                strategy=self.name,
                error=str(e),
            )
            return signals

    def _is_signal_valid_for_regime_enhanced(self, signal: Signal, regime: MarketRegime) -> bool:
        """Enhanced regime-specific validation for breakout signals."""
        try:
            # Get signal metadata
            consolidation_score = signal.metadata.get("consolidation_score", 0.0)
            atr_value = signal.metadata.get("atr", 0.0)
            volume_confirmation = signal.metadata.get("volume_confirmation", False)
            bb_squeeze = signal.metadata.get("bb_squeeze", False)

            # Enhanced regime-specific validation rules
            if regime == MarketRegime.LOW_VOLATILITY:
                # Require higher consolidation scores and ATR values in low volatility
                return consolidation_score > 0.6 and atr_value > 0.005

            elif regime == MarketRegime.HIGH_VOLATILITY:
                # More lenient consolidation requirements but need volume confirmation
                return consolidation_score > 0.3 and (volume_confirmation or bb_squeeze)

            elif regime == MarketRegime.CRISIS:
                # Very strict filtering in crisis conditions
                return (
                    consolidation_score > 0.8
                    and atr_value > 0.01
                    and volume_confirmation
                    and bb_squeeze
                )

            elif regime == MarketRegime.RANGING:
                # Good consolidation essential for range breakouts
                return consolidation_score > 0.7

            elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                # Trend continuation breakouts - moderate requirements
                return consolidation_score > 0.5 and atr_value > 0.003

            else:
                # Standard filtering for other regimes
                return consolidation_score > 0.5

        except Exception as e:
            self.logger.error(
                "Enhanced regime validation failed",
                strategy=self.name,
                symbol=signal.symbol,
                error=str(e),
            )
            return False

    async def _apply_enhanced_time_decay(self, signals: list[Signal], symbol: str) -> list[Signal]:
        """Apply enhanced time-decay adjustments to breakout signals."""
        try:
            last_breakout_times = self._strategy_state["last_breakout_times"]

            if symbol not in last_breakout_times:
                return signals

            last_breakout = last_breakout_times[symbol]
            time_since_breakout = (datetime.now() - last_breakout).total_seconds() / 3600  # Hours

            # Enhanced time decay calculation
            base_decay = self.time_decay_factor**time_since_breakout

            # Additional factors
            consolidation_bonus = 1.0
            if signals:
                avg_consolidation = np.mean(
                    [s.metadata.get("consolidation_score", 0.0) for s in signals]
                )
                consolidation_bonus = 1.0 + (
                    avg_consolidation * 0.2
                )  # Up to 20% bonus for good consolidation

            # Volume bonus
            volume_bonus = 1.0
            if signals:
                avg_volume_conf = np.mean(
                    [1.0 if s.metadata.get("volume_confirmation", False) else 0.8 for s in signals]
                )
                volume_bonus = avg_volume_conf

            effective_decay = base_decay * consolidation_bonus * volume_bonus

            adjusted_signals = []
            for signal in signals:
                # Apply time decay
                original_confidence = signal.confidence
                signal.confidence = signal.confidence * effective_decay

                # Enhanced metadata
                signal.metadata.update(
                    {
                        "time_decay_factor": effective_decay,
                        "time_since_last_breakout_hours": time_since_breakout,
                        "consolidation_bonus": consolidation_bonus,
                        "volume_bonus": volume_bonus,
                        "original_confidence": original_confidence,
                    }
                )

                # Only keep signals with sufficient confidence after decay
                if signal.confidence >= self.config.min_confidence:
                    adjusted_signals.append(signal)

            self.logger.debug(
                "Applied enhanced time decay adjustment",
                strategy=self.name,
                symbol=symbol,
                original_count=len(signals),
                adjusted_count=len(adjusted_signals),
                effective_decay=effective_decay,
                time_since_breakout_hours=time_since_breakout,
            )

            return adjusted_signals

        except Exception as e:
            self.logger.error(
                "Enhanced time decay application failed",
                strategy=self.name,
                symbol=symbol,
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
            # Update ATR values
            if "atr" in indicators:
                self._strategy_state["atr_values"][symbol] = indicators["atr"]

            # Update consolidation scores
            if "consolidation_score" in indicators:
                self._strategy_state["consolidation_scores"][symbol] = {
                    "score": indicators["consolidation_score"],
                    "timestamp": datetime.now(),
                }

            # Update performance metrics
            self._strategy_state["performance_metrics"]["total_breakout_signals"] += len(signals)

            if signals:
                # Update average consolidation score
                current_avg = self._strategy_state["performance_metrics"]["avg_consolidation_score"]
                signals_avg = np.mean([s.metadata.get("consolidation_score", 0.0) for s in signals])
                total_signals = self._strategy_state["performance_metrics"][
                    "total_breakout_signals"
                ]

                if total_signals > 1:
                    new_avg = (
                        (current_avg * (total_signals - len(signals)))
                        + (signals_avg * len(signals))
                    ) / total_signals
                else:
                    new_avg = signals_avg

                self._strategy_state["performance_metrics"]["avg_consolidation_score"] = new_avg

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

    @retry(max_attempts=2, base_delay=1)
    async def validate_signal(self, signal: Signal) -> bool:
        """Enhanced signal validation with service layer integration."""
        try:
            # Basic signal validation
            if not signal or not signal.symbol or not signal.direction:
                return False

            # Enhanced strategy-specific validation
            symbol = signal.symbol

            # Check ATR consistency
            current_atr = self._strategy_state["atr_values"].get(symbol)
            signal_atr = signal.metadata.get("atr", 0.0)

            if current_atr is not None:
                atr_tolerance = 0.3  # 30% tolerance for ATR changes
                if abs(current_atr - signal_atr) / max(current_atr, 0.001) > atr_tolerance:
                    self.logger.warning(
                        "ATR inconsistency detected",
                        strategy=self.name,
                        symbol=symbol,
                        current_atr=current_atr,
                        signal_atr=signal_atr,
                        tolerance=atr_tolerance,
                    )
                    return False

            # Check consolidation score consistency
            consolidation_data = self._strategy_state["consolidation_scores"].get(symbol)
            if consolidation_data:
                current_consolidation = consolidation_data.get("score", 0.0)
                signal_consolidation = signal.metadata.get("consolidation_score", 0.0)

                consolidation_tolerance = 0.4  # 40% tolerance
                if abs(current_consolidation - signal_consolidation) > consolidation_tolerance:
                    self.logger.warning(
                        "Consolidation score inconsistency detected",
                        strategy=self.name,
                        symbol=symbol,
                        current_consolidation=current_consolidation,
                        signal_consolidation=signal_consolidation,
                        tolerance=consolidation_tolerance,
                    )
                    return False

            # Validate signal freshness (within last 5 minutes)
            signal_age = (datetime.now() - signal.timestamp).total_seconds()
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

            # Validate breakout levels make sense
            breakout_type = signal.metadata.get("breakout_type")
            current_price = signal.metadata.get("current_price", 0.0)

            if breakout_type == "upper":
                upper_level = signal.metadata.get("upper_breakout", 0.0)
                if current_price <= upper_level:
                    self.logger.warning(
                        "Invalid upper breakout signal - price not above breakout level",
                        strategy=self.name,
                        symbol=symbol,
                        current_price=current_price,
                        breakout_level=upper_level,
                    )
                    return False
            elif breakout_type == "lower":
                lower_level = signal.metadata.get("lower_breakout", 0.0)
                if current_price >= lower_level:
                    self.logger.warning(
                        "Invalid lower breakout signal - price not below breakout level",
                        strategy=self.name,
                        symbol=symbol,
                        current_price=current_price,
                        breakout_level=lower_level,
                    )
                    return False

            self.logger.debug(
                "Enhanced breakout signal validated",
                strategy=self.name,
                symbol=symbol,
                direction=signal.direction.value,
                confidence=signal.confidence,
                breakout_type=breakout_type,
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
        """Calculate position size using enhanced risk management for breakouts."""
        try:
            # Use adaptive risk manager if available
            if self.adaptive_risk_manager:
                # Get current regime from signal metadata
                regime_str = signal.metadata.get("regime", "medium_volatility")
                try:
                    current_regime = MarketRegime(regime_str)
                except ValueError:
                    current_regime = MarketRegime.MEDIUM_VOLATILITY

                # Get portfolio value
                portfolio_value = Decimal("10000")  # TODO: Get from portfolio service
                if self._risk_manager:
                    try:
                        portfolio_value = self._risk_manager.get_available_capital()
                    except Exception:
                        pass

                # Calculate position size with breakout-specific factors
                atr_value = signal.metadata.get("atr", 0.02)
                consolidation_score = signal.metadata.get("consolidation_score", 0.5)
                volume_confirmation = signal.metadata.get("volume_confirmation", False)

                # Adjust position size based on breakout quality
                base_position_pct = self.config.position_size_pct

                # ATR adjustment (higher ATR = more volatile, smaller position)
                atr_adjustment = max(0.5, 1.0 - (atr_value * 50))

                # Consolidation quality adjustment (better consolidation = larger position)
                consolidation_adjustment = 0.7 + (consolidation_score * 0.6)  # 0.7 to 1.3 range

                # Volume confirmation adjustment
                volume_adjustment = 1.2 if volume_confirmation else 0.9

                # Regime-specific adjustment
                regime_adjustment = self._get_regime_position_adjustment(current_regime)

                adjusted_position_pct = (
                    base_position_pct
                    * atr_adjustment
                    * consolidation_adjustment
                    * volume_adjustment
                    * regime_adjustment
                )

                position_size = portfolio_value * Decimal(str(adjusted_position_pct))

                self.logger.info(
                    "Enhanced breakout position sizing calculated",
                    strategy=self.name,
                    symbol=signal.symbol,
                    position_size=float(position_size),
                    regime=current_regime.value,
                    adjustments={
                        "atr": atr_adjustment,
                        "consolidation": consolidation_adjustment,
                        "volume": volume_adjustment,
                        "regime": regime_adjustment,
                    },
                )

                return position_size
            else:
                # Enhanced fallback calculation
                base_size = Decimal(str(self.config.position_size_pct))
                confidence_adjustment = Decimal(str(signal.confidence))
                consolidation_score = signal.metadata.get("consolidation_score", 0.5)
                consolidation_adjustment = Decimal(str(0.8 + consolidation_score * 0.4))

                adjusted_size = base_size * confidence_adjustment * consolidation_adjustment

                self.logger.info(
                    "Enhanced fallback breakout position sizing",
                    strategy=self.name,
                    symbol=signal.symbol,
                    position_size=float(adjusted_size),
                    confidence_adjustment=float(confidence_adjustment),
                    consolidation_adjustment=float(consolidation_adjustment),
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

    def _get_regime_position_adjustment(self, regime: MarketRegime) -> float:
        """Get regime-specific position size adjustments for breakout strategies."""
        regime_adjustments = {
            MarketRegime.LOW_VOLATILITY: 1.3,  # Larger positions in stable conditions
            MarketRegime.MEDIUM_VOLATILITY: 1.0,  # Standard position sizes
            MarketRegime.HIGH_VOLATILITY: 0.6,  # Smaller positions in volatile conditions
            MarketRegime.CRISIS: 0.4,  # Very small positions in crisis
            MarketRegime.TRENDING_UP: 1.1,  # Slightly larger in uptrend
            MarketRegime.TRENDING_DOWN: 0.8,  # Smaller in downtrend
            MarketRegime.RANGING: 1.2,  # Larger for range breakouts
            MarketRegime.HIGH_CORRELATION: 0.7,  # Smaller when markets move together
            MarketRegime.LOW_CORRELATION: 1.1,  # Larger when markets diverge
        }

        return regime_adjustments.get(regime, 1.0)

    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Enhanced exit logic for breakout positions."""
        try:
            symbol = position.symbol

            # Check for consolidation breakdown
            consolidation_data = self._strategy_state["consolidation_scores"].get(symbol)
            if consolidation_data:
                current_consolidation = consolidation_data.get("score", 0.0)
                entry_consolidation = position.metadata.get("entry_consolidation", 0.0)

                # Exit if consolidation has broken down significantly
                if current_consolidation < entry_consolidation * 0.4:  # More sensitive threshold
                    self.logger.info(
                        "Enhanced consolidation breakdown exit triggered",
                        strategy=self.name,
                        symbol=symbol,
                        entry_consolidation=entry_consolidation,
                        current_consolidation=current_consolidation,
                    )
                    return True

            # Check for ATR expansion (volatility spike)
            current_atr = self._strategy_state["atr_values"].get(symbol)
            if current_atr:
                entry_atr = position.metadata.get("entry_atr", 0.0)

                # Exit if ATR increased significantly (volatility spike = potential reversal)
                if current_atr > entry_atr * 1.8:  # More sensitive threshold
                    self.logger.info(
                        "Enhanced ATR expansion exit triggered",
                        strategy=self.name,
                        symbol=symbol,
                        entry_atr=entry_atr,
                        current_atr=current_atr,
                    )
                    return True

            # Check for regime change exits
            if self.regime_detector:
                try:
                    regime_history = self._strategy_state["regime_history"].get(symbol, [])
                    if regime_history:
                        recent_regime = regime_history[-1]["regime"]
                        entry_regime = position.metadata.get("entry_regime")

                        # Exit if regime has changed to unfavorable conditions
                        unfavorable_regimes = [MarketRegime.CRISIS, MarketRegime.HIGH_VOLATILITY]
                        if (
                            entry_regime
                            and recent_regime != entry_regime
                            and recent_regime in unfavorable_regimes
                        ):
                            self.logger.info(
                                "Enhanced regime change exit triggered",
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
                "atr_values_count": len(self._strategy_state["atr_values"]),
                "breakout_levels_count": len(self._strategy_state["breakout_levels"]),
                "consolidation_scores_count": len(self._strategy_state["consolidation_scores"]),
                "last_breakout_times_count": len(self._strategy_state["last_breakout_times"]),
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
                "atr_period": self.atr_period,
                "breakout_multiplier": self.breakout_multiplier,
                "consolidation_period": self.consolidation_period,
                "volume_confirmation": self.volume_confirmation,
                "time_decay_factor": self.time_decay_factor,
                "breakout_cooldown_minutes": self.breakout_cooldown_minutes,
            },
            "refactoring_status": {
                "centralized_indicators": True,
                "service_layer_integration": True,
                "state_persistence": True,
                "enhanced_metrics": True,
                "removed_direct_access": True,
                "enhanced_consolidation_detection": True,
                "bollinger_band_integration": True,
                "regime_aware_breakouts": True,
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
            "Enhanced volatility breakout strategy started",
            strategy=self.name,
            version="2.0.0",
        )

    async def _on_stop(self) -> None:
        """Enhanced strategy shutdown with state persistence."""
        try:
            # Persist final state
            await self._persist_strategy_state()

            self.logger.info(
                "Enhanced volatility breakout strategy stopped",
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
                "atr_values": {},
                "breakout_levels": {},
                "consolidation_scores": {},
                "last_breakout_times": {},
                "regime_history": {},
                "performance_metrics": {
                    "total_breakout_signals": 0,
                    "successful_breakouts": 0,
                    "false_breakouts": 0,
                    "avg_consolidation_score": 0.0,
                    "regime_transitions": 0,
                },
            }

            self.logger.info(
                "Enhanced volatility breakout strategy cleanup completed",
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
