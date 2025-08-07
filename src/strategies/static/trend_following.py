"""
Trend Following Strategy Implementation.

This module implements a momentum-based trend following strategy that identifies
and follows market trends using moving average crossovers, RSI confirmation,
and volume analysis.

CRITICAL: This strategy MUST inherit from BaseStrategy and follow the exact interface.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio

# MANDATORY: Import from P-011 - NEVER recreate the base strategy
from src.strategies.base import BaseStrategy

# From P-001 - Use existing types  
from src.core.types import (
    Signal, MarketData, Position, SignalDirection,
    StrategyConfig, StrategyType
)
from src.core.exceptions import ValidationError

# From P-007A - Use decorators and validators
from src.utils.decorators import time_execution, retry
from src.utils.validators import validate_price, validate_quantity
from src.utils.helpers import calculate_rsi, calculate_moving_average

# From P-008+ - Use risk management
from src.risk_management.base import BaseRiskManager

# From P-001 - Use structured logging
from src.core.logging import get_logger

logger = get_logger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy Implementation.
    
    This strategy identifies and follows market trends using momentum indicators
    and moving average crossovers. It enters positions when trends are confirmed
    and uses trailing stops to maximize profits.
    
    Key Features:
    - Moving average crossover signals (fast/slow MA)
    - RSI momentum confirmation
    - Volume confirmation requirements
    - Pyramiding support (max 3 levels)
    - Trailing stop implementation
    - Time-based exit rules
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Trend Following Strategy.
        
        Args:
            config: Strategy configuration dictionary
        """
        super().__init__(config)
        # Use the name from config (already set by BaseStrategy)
        self.strategy_type = StrategyType.STATIC
        
        # Strategy-specific parameters with defaults
        self.fast_ma = self.config.parameters.get("fast_ma", 20)
        self.slow_ma = self.config.parameters.get("slow_ma", 50)
        self.rsi_period = self.config.parameters.get("rsi_period", 14)
        self.rsi_overbought = self.config.parameters.get("rsi_overbought", 70)
        self.rsi_oversold = self.config.parameters.get("rsi_oversold", 30)
        self.volume_confirmation = self.config.parameters.get("volume_confirmation", True)
        self.min_volume_ratio = self.config.parameters.get("min_volume_ratio", 1.2)
        self.max_pyramid_levels = self.config.parameters.get("max_pyramid_levels", 3)
        self.trailing_stop_pct = self.config.parameters.get("trailing_stop_pct", 0.02)
        self.time_exit_hours = self.config.parameters.get("time_exit_hours", 48)
        
        # Price history for calculations
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.high_history: List[float] = []
        self.low_history: List[float] = []
        
        # Position tracking for pyramiding
        self.position_levels: Dict[str, int] = {}
        self.position_entries: Dict[str, List[datetime]] = {}
        
        logger.info(
            "Trend Following Strategy initialized",
            strategy=self.name,
            fast_ma=self.fast_ma,
            slow_ma=self.slow_ma,
            rsi_period=self.rsi_period
        )
    
    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> List[Signal]:
        """Generate trend following signals from market data.
        
        MANDATORY: Use graceful error handling and input validation.
        
        Args:
            data: Market data for signal generation
            
        Returns:
            List of trading signals
        """
        try:
            # MANDATORY: Input validation
            if not data or not data.price:
                logger.warning("Invalid market data received", strategy=self.name)
                return []
            
            # Validate price data
            price = float(data.price)
            if price <= 0:
                logger.warning("Invalid price value", strategy=self.name, price=price)
                return []
            
            # Update price history with current data
            self._update_price_history(data)
            
            # Check if we have enough data for calculations
            min_required = max(self.slow_ma, self.rsi_period) + 10
            if len(self.price_history) < min_required:
                logger.debug(
                    "Insufficient price history for signal generation",
                    strategy=self.name,
                    current_length=len(self.price_history),
                    required_length=min_required
                )
                return []
            
            # Calculate technical indicators
            fast_ma = self._calculate_fast_ma()
            slow_ma = self._calculate_slow_ma()
            rsi = self._calculate_rsi()
            
            if fast_ma is None or slow_ma is None or rsi is None:
                return []
            
            # Check volume confirmation if enabled
            if self.volume_confirmation and not self._check_volume_confirmation(data):
                logger.debug(
                    "Volume confirmation failed",
                    strategy=self.name,
                    rsi=rsi,
                    fast_ma=fast_ma,
                    slow_ma=slow_ma
                )
                return []
            
            # Generate signals based on trend analysis
            signals = []
            
            # Trend direction determination
            trend_up = fast_ma > slow_ma
            trend_down = fast_ma < slow_ma
            
            # RSI conditions
            rsi_bullish = rsi > 50 and rsi < self.rsi_overbought
            rsi_bearish = rsi < 50 and rsi > self.rsi_oversold
            
            # Entry signals
            if trend_up and rsi_bullish:
                # Bullish trend signal
                signal = await self._generate_bullish_signal(data, fast_ma, slow_ma, rsi)
                if signal:
                    signals.append(signal)
            
            elif trend_down and rsi_bearish:
                # Bearish trend signal
                signal = await self._generate_bearish_signal(data, fast_ma, slow_ma, rsi)
                if signal:
                    signals.append(signal)
            
            # Exit signals for trend reversal
            if trend_up and rsi_bearish:
                # Exit bullish positions
                signal = await self._generate_exit_signal(data, SignalDirection.SELL, "trend_reversal")
                if signal:
                    signals.append(signal)
            
            elif trend_down and rsi_bullish:
                # Exit bearish positions
                signal = await self._generate_exit_signal(data, SignalDirection.BUY, "trend_reversal")
                if signal:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(
                "Signal generation failed",
                strategy=self.name,
                error=str(e),
                symbol=data.symbol if data else "unknown"
            )
            return []  # MANDATORY: Graceful degradation
    
    def _update_price_history(self, data: MarketData) -> None:
        """Update price and volume history for calculations.
        
        Args:
            data: Market data to add to history
        """
        price = float(data.price)
        volume = float(data.volume) if data.volume else 0.0
        high = float(data.high_price) if data.high_price else price
        low = float(data.low_price) if data.low_price else price
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.high_history.append(high)
        self.low_history.append(low)
        
        # Keep only the required history length
        max_length = max(self.slow_ma, self.rsi_period) + 50
        if len(self.price_history) > max_length:
            self.price_history = self.price_history[-max_length:]
            self.volume_history = self.volume_history[-max_length:]
            self.high_history = self.high_history[-max_length:]
            self.low_history = self.low_history[-max_length:]
    
    def _calculate_fast_ma(self) -> Optional[float]:
        """Calculate fast moving average.
        
        Returns:
            Fast moving average value or None if calculation fails
        """
        try:
            if len(self.price_history) < self.fast_ma:
                return None
            
            recent_prices = self.price_history[-self.fast_ma:]
            return np.mean(recent_prices)
            
        except Exception as e:
            logger.error("Fast MA calculation failed", strategy=self.name, error=str(e))
            return None
    
    def _calculate_slow_ma(self) -> Optional[float]:
        """Calculate slow moving average.
        
        Returns:
            Slow moving average value or None if calculation fails
        """
        try:
            if len(self.price_history) < self.slow_ma:
                return None
            
            recent_prices = self.price_history[-self.slow_ma:]
            return np.mean(recent_prices)
            
        except Exception as e:
            logger.error("Slow MA calculation failed", strategy=self.name, error=str(e))
            return None
    
    def _calculate_rsi(self) -> Optional[float]:
        """Calculate RSI indicator.
        
        Returns:
            RSI value or None if calculation fails
        """
        try:
            if len(self.price_history) < self.rsi_period + 1:
                return None
            
            # Calculate RSI using helper function with price data
            rsi = calculate_rsi(self.price_history, self.rsi_period)
            return rsi
            
        except Exception as e:
            logger.error("RSI calculation failed", strategy=self.name, error=str(e))
            return None
    
    def _check_volume_confirmation(self, data: MarketData) -> bool:
        """Check if volume confirms the trend.
        
        Args:
            data: Current market data
            
        Returns:
            True if volume confirms trend, False otherwise
        """
        try:
            if len(self.volume_history) < self.fast_ma:
                return True  # Pass if insufficient data
            
            current_volume = float(data.volume) if data.volume else 0.0
            if current_volume <= 0:
                return False
            
            # Calculate average volume
            recent_volumes = self.volume_history[-self.fast_ma:]
            avg_volume = np.mean(recent_volumes)
            
            if avg_volume <= 0:
                return True  # Pass if no historical volume data
            
            # Check if current volume is above threshold
            volume_ratio = current_volume / avg_volume
            return volume_ratio >= self.min_volume_ratio
            
        except Exception as e:
            logger.error("Volume confirmation check failed", strategy=self.name, error=str(e))
            return True  # Pass on error
    
    async def _generate_bullish_signal(self, data: MarketData, fast_ma: float, slow_ma: float, rsi: float) -> Optional[Signal]:
        """Generate bullish trend signal.
        
        Args:
            data: Market data
            fast_ma: Fast moving average
            slow_ma: Slow moving average
            rsi: RSI value
            
        Returns:
            Bullish signal or None if validation fails
        """
        try:
            # Check pyramiding limits
            current_levels = self.position_levels.get(data.symbol, 0)
            if current_levels >= self.max_pyramid_levels:
                logger.debug(
                    "Maximum pyramid levels reached",
                    strategy=self.name,
                    symbol=data.symbol,
                    current_levels=current_levels,
                    max_levels=self.max_pyramid_levels
                )
                return None
            
            # Calculate signal confidence
            ma_strength = (fast_ma - slow_ma) / slow_ma
            rsi_strength = (rsi - 50) / 50  # Normalize RSI to -1 to 1
            confidence = min(abs(ma_strength) + abs(rsi_strength), 1.0)
            
            # Ensure minimum confidence for trend signals
            confidence = max(confidence, 0.6)
            
            signal = Signal(
                direction=SignalDirection.BUY,
                confidence=confidence,
                timestamp=data.timestamp,
                symbol=data.symbol,
                strategy_name=self.name,
                metadata={
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "rsi": rsi,
                    "ma_strength": ma_strength,
                    "rsi_strength": rsi_strength,
                    "signal_type": "trend_entry",
                    "trend_direction": "bullish",
                    "pyramid_level": current_levels + 1
                }
            )
            
            if await self.validate_signal(signal):
                logger.info(
                    "Bullish trend signal generated",
                    strategy=self.name,
                    symbol=data.symbol,
                    fast_ma=fast_ma,
                    slow_ma=slow_ma,
                    rsi=rsi,
                    confidence=confidence
                )
                return signal
            
            return None
            
        except Exception as e:
            logger.error("Bullish signal generation failed", strategy=self.name, error=str(e))
            return None
    
    async def _generate_bearish_signal(self, data: MarketData, fast_ma: float, slow_ma: float, rsi: float) -> Optional[Signal]:
        """Generate bearish trend signal.
        
        Args:
            data: Market data
            fast_ma: Fast moving average
            slow_ma: Slow moving average
            rsi: RSI value
            
        Returns:
            Bearish signal or None if validation fails
        """
        try:
            # Check pyramiding limits
            current_levels = self.position_levels.get(data.symbol, 0)
            if current_levels >= self.max_pyramid_levels:
                logger.debug(
                    "Maximum pyramid levels reached",
                    strategy=self.name,
                    symbol=data.symbol,
                    current_levels=current_levels,
                    max_levels=self.max_pyramid_levels
                )
                return None
            
            # Calculate signal confidence
            ma_strength = (slow_ma - fast_ma) / slow_ma
            rsi_strength = (50 - rsi) / 50  # Normalize RSI to -1 to 1
            confidence = min(abs(ma_strength) + abs(rsi_strength), 1.0)
            
            # Ensure minimum confidence for trend signals
            confidence = max(confidence, 0.6)
            
            signal = Signal(
                direction=SignalDirection.SELL,
                confidence=confidence,
                timestamp=data.timestamp,
                symbol=data.symbol,
                strategy_name=self.name,
                metadata={
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "rsi": rsi,
                    "ma_strength": ma_strength,
                    "rsi_strength": rsi_strength,
                    "signal_type": "trend_entry",
                    "trend_direction": "bearish",
                    "pyramid_level": current_levels + 1
                }
            )
            
            if await self.validate_signal(signal):
                logger.info(
                    "Bearish trend signal generated",
                    strategy=self.name,
                    symbol=data.symbol,
                    fast_ma=fast_ma,
                    slow_ma=slow_ma,
                    rsi=rsi,
                    confidence=confidence
                )
                return signal
            
            return None
            
        except Exception as e:
            logger.error("Bearish signal generation failed", strategy=self.name, error=str(e))
            return None
    
    async def _generate_exit_signal(self, data: MarketData, direction: SignalDirection, reason: str) -> Optional[Signal]:
        """Generate exit signal for trend reversal.
        
        Args:
            data: Market data
            direction: Exit direction
            reason: Exit reason
            
        Returns:
            Exit signal or None if validation fails
        """
        try:
            signal = Signal(
                direction=direction,
                confidence=0.8,  # High confidence for exits
                timestamp=data.timestamp,
                symbol=data.symbol,
                strategy_name=self.name,
                metadata={
                    "signal_type": "trend_exit",
                    "exit_reason": reason,
                    "fast_ma": self._calculate_fast_ma(),
                    "slow_ma": self._calculate_slow_ma(),
                    "rsi": self._calculate_rsi()
                }
            )
            
            if await self.validate_signal(signal):
                logger.info(
                    "Trend exit signal generated",
                    strategy=self.name,
                    symbol=data.symbol,
                    direction=direction.value,
                    reason=reason
                )
                return signal
            
            return None
            
        except Exception as e:
            logger.error("Exit signal generation failed", strategy=self.name, error=str(e))
            return None
    
    async def validate_signal(self, signal: Signal) -> bool:
        """Validate trend following signal before execution.
        
        MANDATORY: Check signal confidence, direction, timestamp
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Basic signal validation
            if not signal or signal.confidence < self.config.min_confidence:
                logger.debug(
                    "Signal confidence below threshold",
                    strategy=self.name,
                    confidence=signal.confidence if signal else 0,
                    min_confidence=self.config.min_confidence
                )
                return False
            
            # Check if signal is too old (more than 10 minutes for trend signals)
            if datetime.now(signal.timestamp.tzinfo) - signal.timestamp > timedelta(minutes=10):
                logger.debug("Signal too old", strategy=self.name, signal_age_minutes=10)
                return False
            
            # Validate signal metadata
            metadata = signal.metadata
            if "signal_type" not in metadata:
                logger.warning("Missing signal_type in signal metadata", strategy=self.name)
                return False
            
            # Additional validation for trend entry signals
            if metadata.get("signal_type") == "trend_entry":
                required_fields = ["fast_ma", "slow_ma", "rsi", "trend_direction"]
                for field in required_fields:
                    if field not in metadata:
                        logger.warning(f"Missing {field} in signal metadata", strategy=self.name)
                        return False
                
                # Validate RSI bounds
                rsi = metadata.get("rsi")
                if rsi is None or rsi < 0 or rsi > 100:
                    logger.warning("Invalid RSI value", strategy=self.name, rsi=rsi)
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Signal validation failed", strategy=self.name, error=str(e))
            return False
    
    def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size for trend following signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Position size as Decimal
        """
        try:
            # Base position size from config
            base_size = Decimal(str(self.config.position_size_pct))
            
            # Adjust based on signal confidence
            confidence_factor = signal.confidence
            
            # Adjust based on trend strength
            metadata = signal.metadata
            ma_strength = abs(metadata.get("ma_strength", 0))
            rsi_strength = abs(metadata.get("rsi_strength", 0))
            trend_strength = min(ma_strength + rsi_strength, 1.0)
            
            # Adjust based on pyramid level (smaller positions for higher levels)
            pyramid_level = metadata.get("pyramid_level", 1)
            pyramid_factor = 1.0 / pyramid_level  # Decrease size with each level
            
            # Calculate final position size
            position_size = base_size * Decimal(str(confidence_factor)) * Decimal(str(trend_strength)) * Decimal(str(pyramid_factor))
            
            # Ensure position size is within limits
            max_size = Decimal(str(self.config.parameters.get("max_position_size_pct", 0.1)))
            position_size = min(position_size, max_size)
            
            logger.debug(
                "Position size calculated",
                strategy=self.name,
                base_size=float(base_size),
                confidence_factor=confidence_factor,
                trend_strength=trend_strength,
                pyramid_factor=pyramid_factor,
                final_size=float(position_size)
            )
            
            return position_size
            
        except Exception as e:
            logger.error("Position size calculation failed", strategy=self.name, error=str(e))
            # Return minimum position size on error
            return Decimal(str(self.config.position_size_pct * 0.5))
    
    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Determine if position should be closed.
        
        Args:
            position: Current position
            data: Current market data
            
        Returns:
            True if position should be closed, False otherwise
        """
        try:
            # Update price history for calculations
            self._update_price_history(data)
            
            # Check time-based exit
            if self._should_exit_by_time(position):
                logger.info(
                    "Time-based exit triggered",
                    strategy=self.name,
                    symbol=position.symbol
                )
                return True
            
            # Check trailing stop
            if self._should_exit_by_trailing_stop(position, data):
                logger.info(
                    "Trailing stop triggered",
                    strategy=self.name,
                    symbol=position.symbol
                )
                return True
            
            # Check trend reversal
            fast_ma = self._calculate_fast_ma()
            slow_ma = self._calculate_slow_ma()
            rsi = self._calculate_rsi()
            
            if fast_ma is None or slow_ma is None or rsi is None:
                return False
            
            # Exit if trend has reversed
            if position.side.value == "buy":
                if fast_ma < slow_ma and rsi < 50:
                    logger.info(
                        "Trend reversal exit triggered",
                        strategy=self.name,
                        symbol=position.symbol,
                        fast_ma=fast_ma,
                        slow_ma=slow_ma,
                        rsi=rsi
                    )
                    return True
            else:  # sell position
                if fast_ma > slow_ma and rsi > 50:
                    logger.info(
                        "Trend reversal exit triggered",
                        strategy=self.name,
                        symbol=position.symbol,
                        fast_ma=fast_ma,
                        slow_ma=slow_ma,
                        rsi=rsi
                    )
                    return True
            
            return False
            
        except Exception as e:
            logger.error("Exit check failed", strategy=self.name, error=str(e))
            return False
    
    def _should_exit_by_time(self, position: Position) -> bool:
        """Check if position should be closed due to time limit.
        
        Args:
            position: Current position
            
        Returns:
            True if time limit exceeded, False otherwise
        """
        try:
            # Get position entry time
            entry_time = position.timestamp
            
            # Check if position has been open too long
            current_time = datetime.now(entry_time.tzinfo)
            time_diff = current_time - entry_time
            
            return time_diff > timedelta(hours=self.time_exit_hours)
            
        except Exception as e:
            logger.error("Time exit check failed", strategy=self.name, error=str(e))
            return False
    
    def _should_exit_by_trailing_stop(self, position: Position, data: MarketData) -> bool:
        """Check if position should be closed due to trailing stop.
        
        Args:
            position: Current position
            data: Current market data
            
        Returns:
            True if trailing stop triggered, False otherwise
        """
        try:
            current_price = float(data.price)
            entry_price = float(position.entry_price)
            
            if position.side.value == "buy":
                # For long positions, exit if price falls below trailing stop
                # Trailing stop is entry_price - (entry_price * trailing_stop_pct)
                trailing_stop = entry_price - (entry_price * self.trailing_stop_pct)
                return current_price <= trailing_stop
            else:
                # For short positions, exit if price rises above trailing stop
                # Trailing stop is entry_price + (entry_price * trailing_stop_pct)
                trailing_stop = entry_price + (entry_price * self.trailing_stop_pct)
                return current_price >= trailing_stop
            
        except Exception as e:
            logger.error("Trailing stop check failed", strategy=self.name, error=str(e))
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get trend following strategy information.
        
        Returns:
            Strategy information dictionary
        """
        base_info = super().get_strategy_info()
        
        # Add strategy-specific information
        strategy_info = {
            **base_info,
            "strategy_type": "trend_following",
            "parameters": {
                "fast_ma": self.fast_ma,
                "slow_ma": self.slow_ma,
                "rsi_period": self.rsi_period,
                "rsi_overbought": self.rsi_overbought,
                "rsi_oversold": self.rsi_oversold,
                "volume_confirmation": self.volume_confirmation,
                "min_volume_ratio": self.min_volume_ratio,
                "max_pyramid_levels": self.max_pyramid_levels,
                "trailing_stop_pct": self.trailing_stop_pct,
                "time_exit_hours": self.time_exit_hours
            },
            "price_history_length": len(self.price_history),
            "volume_history_length": len(self.volume_history),
            "position_levels": self.position_levels
        }
        
        return strategy_info 