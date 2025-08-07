"""
Volatility Breakout Strategy Implementation

This module implements a volatility-based breakout strategy that integrates with existing
regime detection and adaptive risk management components.

CRITICAL: This strategy MUST inherit from BaseStrategy and integrate with existing components.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio

# MANDATORY: Import from P-011 - Use exact BaseStrategy interface
from src.strategies.base import BaseStrategy

# MANDATORY: Import from P-010 - Use existing regime detection and adaptive risk
from src.risk_management.regime_detection import MarketRegimeDetector
from src.risk_management.adaptive_risk import AdaptiveRiskManager

# MANDATORY: Import from P-001 - Use existing types
from src.core.types import (
    Signal, MarketData, Position, StrategyType, 
    StrategyConfig, StrategyStatus, SignalDirection, MarketRegime
)
from src.core.exceptions import ValidationError, RiskManagementError
from src.core.logging import get_logger

# MANDATORY: Import from P-007A - Use decorators and validators
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_signal, validate_price, validate_quantity
from src.utils.helpers import calculate_atr, calculate_volatility, calculate_zscore

logger = get_logger(__name__)


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility-based breakout strategy that integrates with existing regime detection and adaptive risk management.
    
    This strategy implements breakout trading using ATR-based thresholds and
    integrates with existing MarketRegimeDetector and AdaptiveRiskManager components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize volatility breakout strategy with configuration."""
        super().__init__(config)
        # Use the name from config (already set by BaseStrategy)
        self.strategy_type = StrategyType.DYNAMIC
        
        # Strategy-specific parameters with defaults from config
        self.atr_period = self.config.parameters.get('atr_period', 14)
        self.breakout_multiplier = self.config.parameters.get('breakout_multiplier', 2.0)
        self.consolidation_period = self.config.parameters.get('consolidation_period', 20)
        self.volume_confirmation = self.config.parameters.get('volume_confirmation', True)
        self.min_consolidation_ratio = self.config.parameters.get('min_consolidation_ratio', 0.8)
        self.max_consolidation_ratio = self.config.parameters.get('max_consolidation_ratio', 1.2)
        self.time_decay_factor = self.config.parameters.get('time_decay_factor', 0.95)
        
        # Integration with existing components
        self.regime_detector: Optional[MarketRegimeDetector] = None
        self.adaptive_risk_manager: Optional[AdaptiveRiskManager] = None
        
        # State tracking for volatility calculations
        self.price_history: Dict[str, List[float]] = {}
        self.high_low_history: Dict[str, List[Tuple[float, float]]] = {}
        self.atr_values: Dict[str, List[float]] = {}
        self.breakout_levels: Dict[str, Dict[str, float]] = {}
        self.consolidation_scores: Dict[str, float] = {}
        self.last_breakout_time: Dict[str, datetime] = {}
        
        logger.info("Volatility breakout strategy initialized", 
                   strategy=self.name,
                   atr_period=self.atr_period,
                   breakout_multiplier=self.breakout_multiplier,
                   consolidation_period=self.consolidation_period)
    
    def set_regime_detector(self, regime_detector: MarketRegimeDetector) -> None:
        """Set the regime detector for adaptive parameter adjustment."""
        self.regime_detector = regime_detector
        logger.info("Regime detector set for volatility breakout strategy", strategy=self.name)
    
    def set_adaptive_risk_manager(self, adaptive_risk_manager: AdaptiveRiskManager) -> None:
        """Set the adaptive risk manager for dynamic risk adjustment."""
        self.adaptive_risk_manager = adaptive_risk_manager
        logger.info("Adaptive risk manager set for volatility breakout strategy", strategy=self.name)
    
    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> List[Signal]:
        """
        Generate volatility breakout signals using existing regime detection.
        
        This implementation:
        1. Uses existing MarketRegimeDetector for regime detection
        2. Calculates ATR-based breakout thresholds
        3. Detects consolidation patterns for breakout setup
        4. Integrates with existing AdaptiveRiskManager for position sizing
        5. Applies time-decay adjustments for breakout signals
        
        Args:
            data: Market data for signal generation
            
        Returns:
            List of trading signals with volatility-aware confidence
        """
        try:
            # MANDATORY: Input validation using utils
            if not validate_price(data.price, data.symbol):
                logger.warning("Invalid market data for signal generation", 
                             strategy=self.name, symbol=data.symbol if data else None)
                return []
            
            # Update price and high-low history
            await self._update_price_history(data)
            
            # Use existing regime detector if available
            current_regime = await self._get_current_regime(data.symbol)
            
            # Calculate volatility indicators using utils
            atr_value = await self._calculate_atr(data.symbol)
            consolidation_score = await self._calculate_consolidation_score(data.symbol)
            breakout_levels = await self._calculate_breakout_levels(data.symbol, atr_value, current_regime)
            
            # Generate signals based on breakout conditions
            signals = await self._generate_breakout_signals(
                data, atr_value, consolidation_score, breakout_levels, current_regime
            )
            
            # Apply regime-specific filtering using existing adaptive risk manager
            signals = await self._apply_regime_filtering(signals, current_regime)
            
            # Apply time-decay adjustments
            signals = await self._apply_time_decay(signals, data.symbol)
            
            logger.info("Generated volatility breakout signals", 
                       strategy=self.name,
                       symbol=data.symbol,
                       signals_count=len(signals),
                       regime=current_regime.value,
                       atr_value=atr_value,
                       consolidation_score=consolidation_score)
            
            return signals
            
        except Exception as e:
            symbol = data.symbol if data else "unknown"
            logger.error("Volatility breakout signal generation failed", 
                        strategy=self.name, symbol=symbol, error=str(e))
            return []  # MANDATORY: Graceful degradation
    
    @retry(max_attempts=3, backoff_factor=2)
    async def _update_price_history(self, data: MarketData) -> None:
        """Update price and high-low history for volatility calculations."""
        symbol = data.symbol
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.high_low_history[symbol] = []
        
        # Add current data point with validation
        if validate_price(data.price, symbol):
            self.price_history[symbol].append(float(data.price))
        else:
            logger.warning("Invalid price data, skipping update", 
                         strategy=self.name, symbol=symbol, price=data.price)
            return
        
        # Calculate high and low for this period
        if data.high_price and data.low_price and validate_price(data.high_price, data.symbol) and validate_price(data.low_price, data.symbol):
            high_low = (float(data.high_price), float(data.low_price))
        else:
            # Use current price as both high and low if not available
            high_low = (float(data.price), float(data.price))
        
        self.high_low_history[symbol].append(high_low)
        
        # Keep only recent history (last 100 points)
        max_history = 100
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
            self.high_low_history[symbol] = self.high_low_history[symbol][-max_history:]
    
    @retry(max_attempts=2)
    async def _get_current_regime(self, symbol: str) -> MarketRegime:
        """Get current market regime using existing regime detector."""
        try:
            if self.regime_detector and symbol in self.price_history:
                # Use existing regime detector's comprehensive regime detection
                market_data = [
                    MarketData(symbol=symbol, price=Decimal(str(p)), volume=Decimal("0"), timestamp=datetime.now())
                    for p in self.price_history[symbol][-20:]  # Last 20 points
                ]
                regime = await self.regime_detector.detect_comprehensive_regime(market_data)
                return regime
            else:
                return MarketRegime.MEDIUM_VOLATILITY  # Default regime
        except Exception as e:
            logger.warning("Regime detection failed, using default", 
                         strategy=self.name, symbol=symbol, error=str(e))
            return MarketRegime.MEDIUM_VOLATILITY
    
    async def _calculate_atr(self, symbol: str) -> float:
        """Calculate Average True Range (ATR) for volatility measurement using utils."""
        try:
            if symbol not in self.high_low_history or len(self.high_low_history[symbol]) < self.atr_period + 1:
                return 0.0
            
            high_lows = self.high_low_history[symbol]
            prices = self.price_history[symbol]
            
            # Calculate True Range
            true_ranges = []
            for i in range(1, len(high_lows)):
                high = high_lows[i][0]
                low = high_lows[i][1]
                prev_close = prices[i-1]
                
                tr1 = high - low  # Current high - current low
                tr2 = abs(high - prev_close)  # Current high - previous close
                tr3 = abs(low - prev_close)   # Current low - previous close
                
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            # Calculate ATR as exponential moving average of true ranges
            if len(true_ranges) >= self.atr_period:
                atr = np.mean(true_ranges[-self.atr_period:])
            else:
                atr = np.mean(true_ranges) if true_ranges else 0.0
            
            # Update ATR history
            if symbol not in self.atr_values:
                self.atr_values[symbol] = []
            self.atr_values[symbol].append(atr)
            
            # Keep only recent ATR values
            if len(self.atr_values[symbol]) > 50:
                self.atr_values[symbol] = self.atr_values[symbol][-50:]
            
            return atr
            
        except Exception as e:
            logger.error("ATR calculation failed", 
                        strategy=self.name, symbol=symbol, error=str(e))
            return 0.0
    
    async def _calculate_consolidation_score(self, symbol: str) -> float:
        """Calculate consolidation score for breakout setup using utils."""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < self.consolidation_period:
                return 0.0
            
            prices = np.array(self.price_history[symbol][-self.consolidation_period:])
            
            # Calculate price range during consolidation period
            price_range = np.max(prices) - np.min(prices)
            avg_price = np.mean(prices)
            
            # Consolidation ratio (range relative to average price)
            consolidation_ratio = price_range / avg_price if avg_price > 0 else 0.0
            
            # Use utils for additional volatility calculation
            price_changes = np.diff(prices)
            volatility = calculate_volatility(price_changes.tolist())
            
            # Score based on consolidation criteria
            if (self.min_consolidation_ratio <= consolidation_ratio <= self.max_consolidation_ratio):
                # Good consolidation - narrow range
                consolidation_score = 1.0 - (consolidation_ratio - self.min_consolidation_ratio) / \
                                   (self.max_consolidation_ratio - self.min_consolidation_ratio)
                
                # Adjust for volatility (lower volatility = higher consolidation score)
                if volatility > 0:
                    volatility_adjustment = 1.0 / (1.0 + volatility)
                    consolidation_score *= volatility_adjustment
            else:
                # Poor consolidation - too wide or too narrow
                consolidation_score = 0.0
            
            self.consolidation_scores[symbol] = consolidation_score
            return consolidation_score
            
        except Exception as e:
            logger.error("Consolidation score calculation failed", 
                        strategy=self.name, symbol=symbol, error=str(e))
            return 0.0
    
    async def _calculate_breakout_levels(self, symbol: str, atr_value: float, 
                                       current_regime: MarketRegime) -> Dict[str, float]:
        """Calculate breakout levels using ATR-based thresholds with regime adjustments."""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 1:
                return {}
            
            current_price = self.price_history[symbol][-1]
            
            # Calculate breakout levels based on ATR
            breakout_distance = atr_value * self.breakout_multiplier
            
            # Apply regime-specific adjustments using existing adaptive risk manager
            regime_adjustment = self._get_regime_breakout_adjustment(current_regime)
            adjusted_distance = breakout_distance * regime_adjustment
            
            breakout_levels = {
                'upper_breakout': current_price + adjusted_distance,
                'lower_breakout': current_price - adjusted_distance,
                'atr_value': atr_value,
                'regime_adjustment': regime_adjustment
            }
            
            self.breakout_levels[symbol] = breakout_levels
            return breakout_levels
            
        except Exception as e:
            logger.error("Breakout levels calculation failed", 
                        strategy=self.name, symbol=symbol, error=str(e))
            return {}
    
    def _get_regime_breakout_adjustment(self, regime: MarketRegime) -> float:
        """Get regime-specific breakout threshold adjustment."""
        regime_adjustments = {
            MarketRegime.LOW_VOLATILITY: 1.5,      # Wider breakout levels in low vol
            MarketRegime.MEDIUM_VOLATILITY: 1.0,    # Standard breakout levels
            MarketRegime.HIGH_VOLATILITY: 0.7,      # Tighter breakout levels in high vol
            MarketRegime.CRISIS: 0.5,               # Very tight levels in crisis
            MarketRegime.TRENDING_UP: 1.2,          # Wider levels in uptrend
            MarketRegime.TRENDING_DOWN: 0.8,        # Tighter levels in downtrend
            MarketRegime.RANGING: 1.1,              # Slightly wider in ranging
            MarketRegime.HIGH_CORRELATION: 0.9,     # Tighter in high correlation
            MarketRegime.LOW_CORRELATION: 1.2       # Wider in low correlation
        }
        
        return regime_adjustments.get(regime, 1.0)
    
    async def _generate_breakout_signals(self, data: MarketData, atr_value: float,
                                       consolidation_score: float, breakout_levels: Dict[str, float],
                                       current_regime: MarketRegime) -> List[Signal]:
        """Generate breakout signals based on price action and volatility."""
        signals = []
        
        try:
            current_price = float(data.price)
            
            # Check for breakout conditions
            if breakout_levels:
                upper_breakout = breakout_levels.get('upper_breakout', 0)
                lower_breakout = breakout_levels.get('lower_breakout', 0)
                
                # Upper breakout (bullish)
                if current_price > upper_breakout and consolidation_score > 0.5:
                    confidence = min(consolidation_score * 0.8, 0.9)
                    
                    signal = Signal(
                        direction=SignalDirection.BUY,
                        confidence=confidence,
                        timestamp=data.timestamp,
                        symbol=data.symbol,
                        strategy_name=self.name,
                        metadata={
                            'atr_value': atr_value,
                            'consolidation_score': consolidation_score,
                            'breakout_level': upper_breakout,
                            'current_price': current_price,
                            'regime': current_regime.value,
                            'breakout_type': 'upper'
                        }
                    )
                    signals.append(signal)
                    
                    # Update last breakout time
                    self.last_breakout_time[data.symbol] = data.timestamp
                
                # Lower breakout (bearish)
                elif current_price < lower_breakout and consolidation_score > 0.5:
                    confidence = min(consolidation_score * 0.8, 0.9)
                    
                    signal = Signal(
                        direction=SignalDirection.SELL,
                        confidence=confidence,
                        timestamp=data.timestamp,
                        symbol=data.symbol,
                        strategy_name=self.name,
                        metadata={
                            'atr_value': atr_value,
                            'consolidation_score': consolidation_score,
                            'breakout_level': lower_breakout,
                            'current_price': current_price,
                            'regime': current_regime.value,
                            'breakout_type': 'lower'
                        }
                    )
                    signals.append(signal)
                    
                    # Update last breakout time
                    self.last_breakout_time[data.symbol] = data.timestamp
            
        except Exception as e:
            logger.error("Breakout signal generation failed", 
                        strategy=self.name, symbol=data.symbol, error=str(e))
        
        return signals
    
    async def _apply_regime_filtering(self, signals: List[Signal], 
                                    current_regime: MarketRegime) -> List[Signal]:
        """Apply regime-specific filtering using existing adaptive risk manager."""
        try:
            if not self.adaptive_risk_manager:
                return signals  # No filtering if adaptive risk manager not available
            
            filtered_signals = []
            
            for signal in signals:
                # Check regime-specific conditions
                if self._is_signal_valid_for_regime(signal, current_regime):
                    # Get adaptive parameters for current regime
                    adaptive_params = self.adaptive_risk_manager.get_adaptive_parameters(current_regime)
                    
                    # Adjust confidence based on regime
                    regime_confidence_multiplier = self._get_regime_confidence_multiplier(current_regime)
                    signal.confidence = min(signal.confidence * regime_confidence_multiplier, 0.95)
                    
                    # Add regime information to metadata
                    signal.metadata.update({
                        'regime_confidence_multiplier': regime_confidence_multiplier,
                        'adaptive_params': adaptive_params
                    })
                    
                    filtered_signals.append(signal)
                else:
                    logger.debug("Signal filtered out by regime conditions", 
                               strategy=self.name,
                               symbol=signal.symbol,
                               regime=current_regime.value)
            
            logger.info("Applied regime filtering", 
                       strategy=self.name,
                       original_count=len(signals),
                       filtered_count=len(filtered_signals),
                       regime=current_regime.value)
            
            return filtered_signals
            
        except Exception as e:
            logger.error("Regime filtering failed", 
                        strategy=self.name, error=str(e))
            return signals  # Return original signals on error
    
    def _is_signal_valid_for_regime(self, signal: Signal, regime: MarketRegime) -> bool:
        """Check if signal is valid for current market regime."""
        try:
            # Get signal metadata
            atr_value = signal.metadata.get('atr_value', 0.0)
            consolidation_score = signal.metadata.get('consolidation_score', 0.0)
            
            # Regime-specific validation rules
            if regime == MarketRegime.LOW_VOLATILITY:
                # Require higher consolidation scores in low volatility
                return consolidation_score > 0.7
            elif regime == MarketRegime.HIGH_VOLATILITY:
                # Allow lower consolidation scores in high volatility
                return consolidation_score > 0.3
            elif regime == MarketRegime.CRISIS:
                # Very strict filtering in crisis
                return consolidation_score > 0.8 and atr_value > 0.01
            else:
                # Standard filtering for other regimes
                return consolidation_score > 0.5
            
        except Exception as e:
            logger.error("Regime validation failed", 
                        strategy=self.name, symbol=signal.symbol, error=str(e))
            return False
    
    def _get_regime_confidence_multiplier(self, regime: MarketRegime) -> float:
        """Get regime-specific confidence multiplier."""
        confidence_multipliers = {
            MarketRegime.LOW_VOLATILITY: 1.1,      # Higher confidence in low vol
            MarketRegime.MEDIUM_VOLATILITY: 1.0,    # Standard confidence
            MarketRegime.HIGH_VOLATILITY: 0.8,      # Lower confidence in high vol
            MarketRegime.CRISIS: 0.6,               # Much lower confidence in crisis
            MarketRegime.TRENDING_UP: 1.05,         # Slightly higher in uptrend
            MarketRegime.TRENDING_DOWN: 0.9,        # Slightly lower in downtrend
            MarketRegime.RANGING: 1.0,              # Standard in ranging
            MarketRegime.HIGH_CORRELATION: 0.9,     # Lower in high correlation
            MarketRegime.LOW_CORRELATION: 1.1       # Higher in low correlation
        }
        
        return confidence_multipliers.get(regime, 1.0)
    
    async def _apply_time_decay(self, signals: List[Signal], symbol: str) -> List[Signal]:
        """Apply time-decay adjustments to breakout signals."""
        try:
            if symbol not in self.last_breakout_time:
                logger.info("No last breakout time found, returning original signals", 
                           strategy=self.name, symbol=symbol)
                return signals
            
            last_breakout = self.last_breakout_time[symbol]
            time_since_breakout = (datetime.now() - last_breakout).total_seconds() / 3600  # Hours
            
            # Apply time decay factor with minimum decay
            decay_factor = max(self.time_decay_factor ** time_since_breakout, 0.8)
            
            adjusted_signals = []
            for signal in signals:
                # Create a copy of the signal to avoid modifying the original
                from copy import deepcopy
                signal_copy = deepcopy(signal)
                
                # Apply time decay to confidence
                original_confidence = signal_copy.confidence
                signal_copy.confidence = signal_copy.confidence * decay_factor
                
                logger.info("Applied time decay", 
                           strategy=self.name,
                           symbol=symbol,
                           original_confidence=original_confidence,
                           new_confidence=signal_copy.confidence,
                           decay_factor=decay_factor,
                           time_since_breakout=time_since_breakout)
                
                # Only keep signals with sufficient confidence after decay
                if signal_copy.confidence >= self.config.min_confidence:
                    adjusted_signals.append(signal_copy)
            
            logger.info("Applied time decay adjustment", 
                       strategy=self.name,
                       original_count=len(signals),
                       adjusted_count=len(adjusted_signals),
                       decay_factor=decay_factor,
                       time_since_breakout=time_since_breakout)
            
            return adjusted_signals
            
        except Exception as e:
            logger.error("Time decay application failed", 
                        strategy=self.name, error=str(e))
            return signals  # Return original signals on error
    
    @retry(max_attempts=2)
    async def validate_signal(self, signal: Signal) -> bool:
        """Validate volatility breakout signal before execution."""
        try:
            # MANDATORY: Basic signal validation using utils
            validate_signal(signal)
            
            # Strategy-specific validation
            symbol = signal.symbol
            
            # Check ATR consistency
            if symbol in self.atr_values:
                current_atr = self.atr_values[symbol][-1] if self.atr_values[symbol] else 0.0
                signal_atr = signal.metadata.get('atr_value', 0.0)
                
                # Validate ATR consistency (within 20% tolerance)
                if abs(current_atr - signal_atr) / max(current_atr, 0.001) > 0.2:
                    logger.warning("ATR inconsistency detected", 
                                 strategy=self.name,
                                 symbol=symbol,
                                 current_atr=current_atr,
                                 signal_atr=signal_atr)
                    return False
            
            # Check consolidation score consistency
            if symbol in self.consolidation_scores:
                current_consolidation = self.consolidation_scores[symbol]
                signal_consolidation = signal.metadata.get('consolidation_score', 0.0)
                
                # Validate consolidation consistency (within 30% tolerance)
                if abs(current_consolidation - signal_consolidation) > 0.3:
                    logger.warning("Consolidation score inconsistency detected", 
                                 strategy=self.name,
                                 symbol=symbol,
                                 current_consolidation=current_consolidation,
                                 signal_consolidation=signal_consolidation)
                    return False
            
            logger.info("Volatility breakout signal validated", 
                       strategy=self.name,
                       symbol=symbol,
                       direction=signal.direction.value,
                       confidence=signal.confidence)
            
            return True
            
        except Exception as e:
            logger.error("Signal validation failed", 
                        strategy=self.name, symbol=signal.symbol, error=str(e))
            return False
    
    async def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size using existing adaptive risk manager."""
        try:
            # Use existing adaptive risk manager if available
            if self.adaptive_risk_manager:
                # Get current regime from signal metadata
                regime_str = signal.metadata.get('regime', 'medium_volatility')
                try:
                    current_regime = MarketRegime(regime_str)
                except ValueError:
                    current_regime = MarketRegime.MEDIUM_VOLATILITY
                
                # Use existing adaptive risk manager for position sizing
                # This integrates with the comprehensive regime detection and adaptive sizing
                portfolio_value = Decimal("10000")  # TODO: Get actual portfolio value
                position_size = await self.adaptive_risk_manager.calculate_adaptive_position_size(
                    signal, current_regime, portfolio_value
                )
                
                logger.info("Used adaptive risk manager for position sizing", 
                           strategy=self.name,
                           symbol=signal.symbol,
                           position_size=float(position_size),
                           regime=current_regime.value)
                
                return position_size
            else:
                # Fallback to base position size
                base_size = Decimal(str(self.config.position_size_pct))
                logger.info("Used fallback position sizing", 
                           strategy=self.name,
                           symbol=signal.symbol,
                           position_size=float(base_size))
                return base_size
            
        except Exception as e:
            logger.error("Position size calculation failed", 
                        strategy=self.name, symbol=signal.symbol, error=str(e))
            return Decimal(str(self.config.position_size_pct))
    
    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Determine if position should be closed based on volatility criteria."""
        try:
            # Check for consolidation breakdown
            if position.symbol in self.consolidation_scores:
                current_consolidation = self.consolidation_scores[position.symbol]
                entry_consolidation = position.metadata.get('entry_consolidation', 0.0)
                
                # Exit if consolidation has broken down significantly
                if current_consolidation < entry_consolidation * 0.5:
                    logger.info("Consolidation breakdown exit triggered", 
                               strategy=self.name,
                               symbol=position.symbol,
                               entry_consolidation=entry_consolidation,
                               current_consolidation=current_consolidation)
                    return True
            
            # Check for ATR expansion (volatility increase)
            if position.symbol in self.atr_values and len(self.atr_values[position.symbol]) >= 1:
                current_atr = self.atr_values[position.symbol][-1]
                entry_atr = position.metadata.get('entry_atr', 0.0)
                
                # Exit if ATR has increased significantly (volatility spike)
                if current_atr > entry_atr * 1.5:
                    logger.info("ATR expansion exit triggered", 
                               strategy=self.name,
                               symbol=position.symbol,
                               entry_atr=entry_atr,
                               current_atr=current_atr)
                    return True
            
            return False
            
        except Exception as e:
            logger.error("Exit condition check failed", 
                        strategy=self.name, symbol=position.symbol, error=str(e))
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information."""
        base_info = super().get_strategy_info()
        
        # Add integration information
        integration_info = {
            'atr_values_count': len(self.atr_values),
            'consolidation_scores': self.consolidation_scores,
            'breakout_levels_count': len(self.breakout_levels),
            'price_history_count': len(self.price_history),
            'regime_detector_available': self.regime_detector is not None,
            'adaptive_risk_manager_available': self.adaptive_risk_manager is not None
        }
        
        base_info.update(integration_info)
        return base_info
