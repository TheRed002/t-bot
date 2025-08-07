"""
Adaptive Momentum Strategy Implementation

This module implements a regime-aware momentum strategy that integrates with existing
regime detection and adaptive risk management components.

CRITICAL: This strategy MUST inherit from BaseStrategy and integrate with existing components.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
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
from src.utils.helpers import calculate_atr, calculate_zscore, calculate_volatility

logger = get_logger(__name__)


class AdaptiveMomentumStrategy(BaseStrategy):
    """
    Adaptive momentum strategy that integrates with existing regime detection and adaptive risk management.
    
    This strategy implements momentum-based trading with dynamic parameter adjustment
    using the existing MarketRegimeDetector and AdaptiveRiskManager components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize adaptive momentum strategy with configuration."""
        super().__init__(config)
        # Use the name from config (already set by BaseStrategy)
        self.strategy_type = StrategyType.DYNAMIC
        
        # Strategy-specific parameters with defaults from config
        self.fast_ma_period = self.config.parameters.get('fast_ma_period', 20)
        self.slow_ma_period = self.config.parameters.get('slow_ma_period', 50)
        self.rsi_period = self.config.parameters.get('rsi_period', 14)
        self.rsi_overbought = self.config.parameters.get('rsi_overbought', 70)
        self.rsi_oversold = self.config.parameters.get('rsi_oversold', 30)
        self.momentum_lookback = self.config.parameters.get('momentum_lookback', 10)
        self.volume_threshold = self.config.parameters.get('volume_threshold', 1.5)
        
        # Integration with existing components
        self.regime_detector: Optional[MarketRegimeDetector] = None
        self.adaptive_risk_manager: Optional[AdaptiveRiskManager] = None
        
        # State tracking for momentum calculations
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.momentum_scores: Dict[str, float] = {}
        
        logger.info("Adaptive momentum strategy initialized", 
                   strategy=self.name,
                   fast_ma=self.fast_ma_period,
                   slow_ma=self.slow_ma_period,
                   rsi_period=self.rsi_period)
    
    def set_regime_detector(self, regime_detector: MarketRegimeDetector) -> None:
        """Set the regime detector for adaptive parameter adjustment."""
        self.regime_detector = regime_detector
        logger.info("Regime detector set for adaptive momentum strategy", strategy=self.name)
    
    def set_adaptive_risk_manager(self, adaptive_risk_manager: AdaptiveRiskManager) -> None:
        """Set the adaptive risk manager for dynamic risk adjustment."""
        self.adaptive_risk_manager = adaptive_risk_manager
        logger.info("Adaptive risk manager set for adaptive momentum strategy", strategy=self.name)
    
    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> List[Signal]:
        """
        Generate adaptive momentum signals using existing regime detection.
        
        This implementation:
        1. Uses existing MarketRegimeDetector for regime detection
        2. Calculates momentum indicators for signal generation
        3. Integrates with existing AdaptiveRiskManager for position sizing
        4. Applies regime-aware confidence adjustments
        
        Args:
            data: Market data for signal generation
            
        Returns:
            List of trading signals with regime-aware confidence
        """
        try:
            # MANDATORY: Input validation using utils
            if not validate_price(data.price, data.symbol):
                logger.warning("Invalid market data for signal generation", 
                             strategy=self.name, symbol=data.symbol if data else None)
                return []
            
            # Update price and volume history
            await self._update_price_history(data)
            
            # Use existing regime detector if available
            current_regime = await self._get_current_regime(data.symbol)
            
            # Calculate momentum indicators using utils
            momentum_score = await self._calculate_momentum_score(data.symbol)
            volume_score = await self._calculate_volume_score(data.symbol)
            rsi_score = await self._calculate_rsi_score(data.symbol)
            
            # Generate signals based on momentum and regime
            signals = await self._generate_momentum_signals(
                data, momentum_score, volume_score, rsi_score, current_regime
            )
            
            # Apply regime-aware confidence adjustments using existing adaptive risk manager
            signals = await self._apply_regime_confidence_adjustments(signals, current_regime)
            
            logger.info("Generated adaptive momentum signals", 
                       strategy=self.name,
                       symbol=data.symbol,
                       signals_count=len(signals),
                       regime=current_regime.value,
                       momentum_score=momentum_score)
            
            return signals
            
        except Exception as e:
            symbol = data.symbol if data else "unknown"
            logger.error("Adaptive momentum signal generation failed", 
                        strategy=self.name, symbol=symbol, error=str(e))
            return []  # MANDATORY: Graceful degradation
    
    @retry(max_attempts=3, backoff_factor=2)
    async def _update_price_history(self, data: MarketData) -> None:
        """Update price and volume history for momentum calculations."""
        symbol = data.symbol
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
        
        # Add current data point with validation
        if validate_price(data.price, symbol):
            self.price_history[symbol].append(float(data.price))
        else:
            logger.warning("Invalid price data, skipping update", 
                         strategy=self.name, symbol=symbol, price=data.price)
            return
            
        if validate_quantity(data.volume, data.symbol):
            self.volume_history[symbol].append(float(data.volume))
        else:
            logger.warning("Invalid volume data, using default", 
                         strategy=self.name, symbol=symbol, volume=data.volume)
            self.volume_history[symbol].append(0.0)
        
        # Keep only recent history (last 100 points)
        max_history = 100
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
            self.volume_history[symbol] = self.volume_history[symbol][-max_history:]
    
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
    
    async def _calculate_momentum_score(self, symbol: str) -> float:
        """Calculate momentum score using multiple indicators and utils."""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < self.slow_ma_period:
                return 0.0
            
            prices = np.array(self.price_history[symbol])
            
            # Calculate moving averages
            fast_ma = np.mean(prices[-self.fast_ma_period:])
            slow_ma = np.mean(prices[-self.slow_ma_period:])
            
            # Calculate momentum indicators
            price_momentum = (prices[-1] - prices[-self.momentum_lookback]) / prices[-self.momentum_lookback]
            ma_momentum = (fast_ma - slow_ma) / slow_ma
            
            # Use utils for additional calculations
            price_changes = np.diff(prices[-self.momentum_lookback:])
            volatility = calculate_volatility(price_changes.tolist())
            z_score = calculate_zscore(prices.tolist(), self.momentum_lookback)
            
            # Combine momentum scores with volatility adjustment
            momentum_score = (price_momentum + ma_momentum) / 2
            
            # Adjust for volatility (higher volatility = lower momentum score)
            if volatility > 0:
                volatility_adjustment = 1.0 / (1.0 + volatility)
                momentum_score *= volatility_adjustment
            
            # Normalize to [-1, 1] range
            momentum_score = np.clip(momentum_score, -1.0, 1.0)
            
            self.momentum_scores[symbol] = momentum_score
            return momentum_score
            
        except Exception as e:
            logger.error("Momentum score calculation failed", 
                        strategy=self.name, symbol=symbol, error=str(e))
            return 0.0
    
    async def _calculate_volume_score(self, symbol: str) -> float:
        """Calculate volume-based momentum score using utils."""
        try:
            if symbol not in self.volume_history or len(self.volume_history[symbol]) < 20:
                return 0.0
            
            volumes = np.array(self.volume_history[symbol])
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[-20:])
            
            # Volume ratio compared to average
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Normalize volume score
            volume_score = min(volume_ratio / self.volume_threshold, 1.0)
            
            return volume_score
            
        except Exception as e:
            logger.error("Volume score calculation failed", 
                        strategy=self.name, symbol=symbol, error=str(e))
            return 0.0
    
    async def _calculate_rsi_score(self, symbol: str) -> float:
        """Calculate RSI-based momentum score using utils."""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < self.rsi_period + 1:
                return 0.0
            
            prices = np.array(self.price_history[symbol])
            
            # Calculate price changes
            price_changes = np.diff(prices)
            
            # Calculate RSI
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)
            
            avg_gain = np.mean(gains[-self.rsi_period:])
            avg_loss = np.mean(losses[-self.rsi_period:])
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Convert RSI to momentum score [-1, 1]
            if rsi > self.rsi_overbought:
                rsi_score = -0.5  # Overbought, potential reversal
            elif rsi < self.rsi_oversold:
                rsi_score = 0.5   # Oversold, potential reversal
            else:
                rsi_score = (rsi - 50) / 50  # Normal range
                
            return rsi_score
            
        except Exception as e:
            logger.error("RSI score calculation failed", 
                        strategy=self.name, symbol=symbol, error=str(e))
            return 0.0
    
    async def _generate_momentum_signals(self, data: MarketData, momentum_score: float,
                                       volume_score: float, rsi_score: float, 
                                       current_regime: MarketRegime) -> List[Signal]:
        """Generate momentum-based trading signals."""
        signals = []
        
        try:
            # Calculate combined momentum score
            combined_score = (momentum_score * 0.5 + volume_score * 0.3 + rsi_score * 0.2)
            
            # Generate signals based on momentum strength
            if combined_score > 0.3:  # Strong positive momentum
                signal = Signal(
                    direction=SignalDirection.BUY,
                    confidence=min(abs(combined_score), 0.9),
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    strategy_name=self.name,
                    metadata={
                        'momentum_score': momentum_score,
                        'volume_score': volume_score,
                        'rsi_score': rsi_score,
                        'regime': current_regime.value,
                        'combined_score': combined_score
                    }
                )
                signals.append(signal)
                
            elif combined_score < -0.3:  # Strong negative momentum
                signal = Signal(
                    direction=SignalDirection.SELL,
                    confidence=min(abs(combined_score), 0.9),
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    strategy_name=self.name,
                    metadata={
                        'momentum_score': momentum_score,
                        'volume_score': volume_score,
                        'rsi_score': rsi_score,
                        'regime': current_regime.value,
                        'combined_score': combined_score
                    }
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error("Momentum signal generation failed", 
                        strategy=self.name, symbol=data.symbol, error=str(e))
        
        return signals
    
    async def _apply_regime_confidence_adjustments(self, signals: List[Signal], 
                                                 current_regime: MarketRegime) -> List[Signal]:
        """Apply regime-aware confidence adjustments using existing adaptive risk manager."""
        try:
            if not self.adaptive_risk_manager:
                return signals  # No adjustments if adaptive risk manager not available
            
            adjusted_signals = []
            for signal in signals:
                # Get adaptive parameters for current regime
                adaptive_params = self.adaptive_risk_manager.get_adaptive_parameters(current_regime)
                
                # Apply regime-specific confidence adjustment
                # Higher confidence in favorable regimes, lower in unfavorable ones
                regime_confidence_multiplier = self._get_regime_confidence_multiplier(current_regime)
                adjusted_confidence = signal.confidence * regime_confidence_multiplier
                
                # Update signal confidence
                signal.confidence = min(adjusted_confidence, 0.95)
                
                # Add regime information to metadata
                signal.metadata.update({
                    'regime_confidence_multiplier': regime_confidence_multiplier,
                    'adaptive_params': adaptive_params
                })
                
                adjusted_signals.append(signal)
            
            logger.info("Applied regime confidence adjustments", 
                       strategy=self.name,
                       original_count=len(signals),
                       adjusted_count=len(adjusted_signals),
                       regime=current_regime.value)
            
            return adjusted_signals
            
        except Exception as e:
            logger.error("Regime confidence adjustment failed", 
                        strategy=self.name, error=str(e))
            return signals  # Return original signals on error
    
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
    
    @retry(max_attempts=2)
    async def validate_signal(self, signal: Signal) -> bool:
        """Validate adaptive momentum signal before execution."""
        try:
            # MANDATORY: Basic signal validation using utils
            validate_signal(signal)
            
            # Strategy-specific validation
            if signal.symbol not in self.momentum_scores:
                logger.warning("No momentum data for signal validation", 
                             strategy=self.name, symbol=signal.symbol)
                return False
            
            # Check momentum score consistency
            momentum_score = self.momentum_scores.get(signal.symbol, 0.0)
            signal_momentum = signal.metadata.get('momentum_score', 0.0)
            
            # Validate momentum consistency (within 20% tolerance)
            if abs(momentum_score - signal_momentum) > 0.2:
                logger.warning("Momentum score inconsistency detected", 
                             strategy=self.name,
                             symbol=signal.symbol,
                             current_momentum=momentum_score,
                             signal_momentum=signal_momentum)
                return False
            
            logger.info("Adaptive momentum signal validated", 
                       strategy=self.name,
                       symbol=signal.symbol,
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
        """Determine if position should be closed based on momentum criteria."""
        try:
            # Check momentum reversal
            if position.symbol in self.momentum_scores:
                current_momentum = self.momentum_scores[position.symbol]
                entry_momentum = position.metadata.get('entry_momentum', 0.0)
                
                # Exit if momentum has reversed significantly
                if (position.side.value == 'buy' and current_momentum < -0.2) or \
                   (position.side.value == 'sell' and current_momentum > 0.2):
                    logger.info("Momentum reversal exit triggered", 
                               strategy=self.name,
                               symbol=position.symbol,
                               entry_momentum=entry_momentum,
                               current_momentum=current_momentum)
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
            'momentum_scores': self.momentum_scores,
            'price_history_count': len(self.price_history),
            'volume_history_count': len(self.volume_history),
            'regime_detector_available': self.regime_detector is not None,
            'adaptive_risk_manager_available': self.adaptive_risk_manager is not None
        }
        
        base_info.update(integration_info)
        return base_info
