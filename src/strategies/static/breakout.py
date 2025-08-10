"""
Breakout Strategy Implementation.

This module implements a breakout strategy that identifies support/resistance
breakouts with volume confirmation and false breakout filtering.

CRITICAL: This strategy MUST inherit from BaseStrategy and follow the exact interface.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np

# From P-001 - Use structured logging
from src.core.logging import get_logger

# From P-001 - Use existing types
from src.core.types import (
    MarketData,
    Position,
    Signal,
    SignalDirection,
    StrategyType,
)

# From P-008+ - Use risk management
# MANDATORY: Import from P-011 - NEVER recreate the base strategy
from src.strategies.base import BaseStrategy

# From P-007A - Use decorators and validators
from src.utils.decorators import time_execution
from src.utils.helpers import calculate_atr

logger = get_logger(__name__)


class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy Implementation.

    This strategy identifies support/resistance breakouts and enters positions
    when price breaks through these levels with volume confirmation.

    Key Features:
    - Support/resistance level detection
    - Breakout confirmation with volume
    - False breakout filtering
    - Consolidation period requirements
    - Dynamic stop loss placement
    - Target calculation based on range
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize Breakout Strategy.

        Args:
            config: Strategy configuration dictionary
        """
        super().__init__(config)
        # Use the name from config (already set by BaseStrategy)
        self.strategy_type = StrategyType.STATIC

        # Strategy-specific parameters with defaults
        self.lookback_period = self.config.parameters.get("lookback_period", 20)
        self.breakout_threshold = self.config.parameters.get("breakout_threshold", 0.02)  # 2%
        self.volume_multiplier = self.config.parameters.get("volume_multiplier", 1.5)
        self.consolidation_periods = self.config.parameters.get("consolidation_periods", 5)
        self.false_breakout_filter = self.config.parameters.get("false_breakout_filter", True)
        self.false_breakout_threshold = self.config.parameters.get(
            "false_breakout_threshold", 0.01
        )  # 1%
        self.target_multiplier = self.config.parameters.get("target_multiplier", 2.0)
        self.atr_period = self.config.parameters.get("atr_period", 14)
        self.atr_multiplier = self.config.parameters.get("atr_multiplier", 2.0)

        # Price history for calculations
        self.price_history: list[float] = []
        self.volume_history: list[float] = []
        self.high_history: list[float] = []
        self.low_history: list[float] = []

        # Support/resistance levels
        self.support_levels: list[float] = []
        self.resistance_levels: list[float] = []
        self.last_breakout_time: datetime | None = None

        logger.info(
            "Breakout Strategy initialized",
            strategy=self.name,
            lookback_period=self.lookback_period,
            breakout_threshold=self.breakout_threshold,
            volume_multiplier=self.volume_multiplier,
        )

    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """Generate breakout signals from market data.

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

            # Update price history
            self._update_price_history(data)

            # Check if we have enough data for calculations
            if len(self.price_history) < self.lookback_period:
                logger.debug(
                    "Insufficient price history for signal generation",
                    strategy=self.name,
                    current_length=len(self.price_history),
                    required_length=self.lookback_period,
                )
                return []

            # Update support/resistance levels
            self._update_support_resistance_levels()

            # Check for consolidation period
            consolidation_met = self._check_consolidation_period()
            logger.debug(
                "Consolidation check result",
                strategy=self.name,
                consolidation_met=consolidation_met,
                required_periods=self.consolidation_periods,
                price_history_length=len(self.price_history),
            )
            if not consolidation_met:
                logger.debug(
                    "Consolidation period not met",
                    strategy=self.name,
                    required_periods=self.consolidation_periods,
                )
                return []

            # Generate signals based on breakout analysis
            signals = []

            # Check for resistance breakout (bullish)
            resistance_breakout = self._check_resistance_breakout(data)
            if resistance_breakout:
                signal = await self._generate_bullish_breakout_signal(data, resistance_breakout)
                if signal:
                    signals.append(signal)

            # Check for support breakout (bearish)
            support_breakout = self._check_support_breakout(data)
            if support_breakout:
                signal = await self._generate_bearish_breakout_signal(data, support_breakout)
                if signal:
                    signals.append(signal)

            # Check for false breakout exits
            false_breakout_exit = self._check_false_breakout(data)
            if false_breakout_exit:
                signal = await self._generate_false_breakout_exit_signal(data, false_breakout_exit)
                if signal:
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(
                "Signal generation failed",
                strategy=self.name,
                error=str(e),
                symbol=data.symbol if data else "unknown",
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
        max_length = max(self.lookback_period, self.atr_period) + 50
        if len(self.price_history) > max_length:
            self.price_history = self.price_history[-max_length:]
            self.volume_history = self.volume_history[-max_length:]
            self.high_history = self.high_history[-max_length:]
            self.low_history = self.low_history[-max_length:]

    def _update_support_resistance_levels(self) -> None:
        """Update support and resistance levels based on recent price action."""
        try:
            if len(self.price_history) < self.lookback_period:
                return

            # Calculate support and resistance levels
            recent_highs = self.high_history[-self.lookback_period :]
            recent_lows = self.low_history[-self.lookback_period :]

            # Find resistance levels (local highs)
            resistance_levels = []
            for i in range(1, len(recent_highs) - 1):
                if recent_highs[i] > recent_highs[i - 1] and recent_highs[i] > recent_highs[i + 1]:
                    resistance_levels.append(recent_highs[i])

            # Find support levels (local lows)
            support_levels = []
            for i in range(1, len(recent_lows) - 1):
                if recent_lows[i] < recent_lows[i - 1] and recent_lows[i] < recent_lows[i + 1]:
                    support_levels.append(recent_lows[i])

            # Update levels (keep only significant ones)
            self.resistance_levels = sorted(list(set(resistance_levels)))
            self.support_levels = sorted(list(set(support_levels)))

        except Exception as e:
            logger.error("Support/resistance update failed", strategy=self.name, error=str(e))

    def _check_consolidation_period(self) -> bool:
        """Check if price has been consolidating for required periods.

        Returns:
            True if consolidation period met, False otherwise
        """
        try:
            # If consolidation_periods is 0, skip the check (always pass)
            if self.consolidation_periods == 0:
                return True

            if len(self.price_history) < self.consolidation_periods:
                return False

            # Check if price has been within a narrow range
            recent_prices = self.price_history[-self.consolidation_periods :]
            price_range = max(recent_prices) - min(recent_prices)
            avg_price = np.mean(recent_prices)

            # Consolidation if range is less than 2% of average price
            consolidation_threshold = avg_price * 0.02
            is_consolidating = price_range <= consolidation_threshold

            logger.debug(
                "Consolidation check details",
                strategy=self.name,
                recent_prices_count=len(recent_prices),
                price_range=price_range,
                avg_price=avg_price,
                consolidation_threshold=consolidation_threshold,
                is_consolidating=is_consolidating,
            )

            return is_consolidating

        except Exception as e:
            logger.error("Consolidation check failed", strategy=self.name, error=str(e))
            return False

    def _check_resistance_breakout(self, data: MarketData) -> dict[str, Any] | None:
        """Check for resistance breakout.

        Args:
            data: Current market data

        Returns:
            Breakout information or None if no breakout
        """
        try:
            current_price = float(data.price)
            current_volume = float(data.volume) if data.volume else 0.0

            logger.debug(
                "Checking resistance breakout",
                strategy=self.name,
                current_price=current_price,
                current_volume=current_volume,
                resistance_levels=self.resistance_levels,
                breakout_threshold=self.breakout_threshold,
            )

            # Check each resistance level
            for resistance in self.resistance_levels:
                breakout_price = resistance * (1 + self.breakout_threshold)
                logger.debug(
                    "Checking resistance level",
                    resistance=resistance,
                    breakout_price=breakout_price,
                    current_price=current_price,
                    is_breakout=current_price > breakout_price,
                )

                # Check if price broke above resistance
                if current_price > breakout_price:
                    # Check volume confirmation
                    volume_confirmed = self._check_volume_confirmation(current_volume)

                    if volume_confirmed:
                        return {
                            "level": resistance,
                            "breakout_price": current_price,
                            "breakout_type": "resistance",
                            "volume": current_volume,
                        }

            return None

        except Exception as e:
            logger.error("Resistance breakout check failed", strategy=self.name, error=str(e))
            return None

    def _check_support_breakout(self, data: MarketData) -> dict[str, Any] | None:
        """Check for support breakout.

        Args:
            data: Current market data

        Returns:
            Breakout information or None if no breakout
        """
        try:
            current_price = float(data.price)
            current_volume = float(data.volume) if data.volume else 0.0

            # Check each support level
            for support in self.support_levels:
                # Check if price broke below support
                if current_price < support * (1 - self.breakout_threshold):
                    # Check volume confirmation
                    if self._check_volume_confirmation(current_volume):
                        return {
                            "level": support,
                            "breakout_price": current_price,
                            "breakout_type": "support",
                            "volume": current_volume,
                        }

            return None

        except Exception as e:
            logger.error("Support breakout check failed", strategy=self.name, error=str(e))
            return None

    def _check_volume_confirmation(self, current_volume: float) -> bool:
        """Check if volume confirms the breakout.

        Args:
            current_volume: Current volume

        Returns:
            True if volume confirms breakout, False otherwise
        """
        try:
            if current_volume <= 0:
                return False

            if len(self.volume_history) < self.lookback_period:
                return True  # Pass if insufficient data

            # Calculate average volume
            recent_volumes = self.volume_history[-self.lookback_period :]
            avg_volume = np.mean(recent_volumes)

            if avg_volume <= 0:
                return True  # Pass if no historical volume data

            # Check if current volume is above threshold
            volume_ratio = current_volume / avg_volume
            return bool(volume_ratio >= self.volume_multiplier)

        except Exception as e:
            logger.error("Volume confirmation check failed", strategy=self.name, error=str(e))
            return True  # Pass on error

    def _check_false_breakout(self, data: MarketData) -> dict[str, Any] | None:
        """Check for false breakout (price returned to level).

        Args:
            data: Current market data

        Returns:
            False breakout information or None
        """
        try:
            if not self.false_breakout_filter:
                return None

            current_price = float(data.price)

            # Check if price returned to resistance level
            for resistance in self.resistance_levels:
                if abs(current_price - resistance) / resistance <= self.false_breakout_threshold:
                    return {
                        "level": resistance,
                        "current_price": current_price,
                        "breakout_type": "false_resistance",
                    }

            # Check if price returned to support level
            for support in self.support_levels:
                if abs(current_price - support) / support <= self.false_breakout_threshold:
                    return {
                        "level": support,
                        "current_price": current_price,
                        "breakout_type": "false_support",
                    }

            return None

        except Exception as e:
            logger.error("False breakout check failed", strategy=self.name, error=str(e))
            return None

    async def _generate_bullish_breakout_signal(
        self, data: MarketData, breakout_info: dict[str, Any]
    ) -> Signal | None:
        """Generate bullish breakout signal.

        Args:
            data: Market data
            breakout_info: Breakout information

        Returns:
            Bullish signal or None if validation fails
        """
        try:
            # Calculate signal confidence
            level = breakout_info["level"]
            breakout_price = breakout_info["breakout_price"]
            volume = breakout_info["volume"]

            # Confidence based on breakout strength and volume
            breakout_strength = (breakout_price - level) / level

            # Calculate volume strength with fallback for insufficient data
            if len(self.volume_history) >= self.lookback_period:
                avg_volume = np.mean(self.volume_history[-self.lookback_period :])
                volume_strength = min(volume / avg_volume, 3.0) if avg_volume > 0 else 1.0
            else:
                volume_strength = 1.0  # Default strength if insufficient data

            confidence = min(breakout_strength + volume_strength, 1.0)

            # Calculate target price
            range_size = level - min(self.support_levels) if self.support_levels else level * 0.1
            target_price = breakout_price + (range_size * self.target_multiplier)

            signal = Signal(
                direction=SignalDirection.BUY,
                confidence=confidence,
                timestamp=data.timestamp,
                symbol=data.symbol,
                strategy_name=self.name,
                metadata={
                    "breakout_level": level,
                    "breakout_price": breakout_price,
                    "target_price": target_price,
                    "volume": volume,
                    "signal_type": "breakout_entry",
                    "breakout_direction": "bullish",
                    "range_size": range_size,
                },
            )

            if await self.validate_signal(signal):
                self.last_breakout_time = data.timestamp
                logger.info(
                    "Bullish breakout signal generated",
                    strategy=self.name,
                    symbol=data.symbol,
                    level=level,
                    breakout_price=breakout_price,
                    confidence=confidence,
                )
                return signal

            return None

        except Exception as e:
            logger.error(
                "Bullish breakout signal generation failed", strategy=self.name, error=str(e)
            )
            return None

    async def _generate_bearish_breakout_signal(
        self, data: MarketData, breakout_info: dict[str, Any]
    ) -> Signal | None:
        """Generate bearish breakout signal.

        Args:
            data: Market data
            breakout_info: Breakout information

        Returns:
            Bearish signal or None if validation fails
        """
        try:
            # Calculate signal confidence
            level = breakout_info["level"]
            breakout_price = breakout_info["breakout_price"]
            volume = breakout_info["volume"]

            # Confidence based on breakout strength and volume
            breakout_strength = (level - breakout_price) / level

            # Calculate volume strength with fallback for insufficient data
            if len(self.volume_history) >= self.lookback_period:
                avg_volume = np.mean(self.volume_history[-self.lookback_period :])
                volume_strength = min(volume / avg_volume, 3.0) if avg_volume > 0 else 1.0
            else:
                volume_strength = 1.0  # Default strength if insufficient data

            confidence = min(breakout_strength + volume_strength, 1.0)

            # Calculate target price
            range_size = (
                max(self.resistance_levels) - level if self.resistance_levels else level * 0.1
            )
            target_price = breakout_price - (range_size * self.target_multiplier)

            signal = Signal(
                direction=SignalDirection.SELL,
                confidence=confidence,
                timestamp=data.timestamp,
                symbol=data.symbol,
                strategy_name=self.name,
                metadata={
                    "breakout_level": level,
                    "breakout_price": breakout_price,
                    "target_price": target_price,
                    "volume": volume,
                    "signal_type": "breakout_entry",
                    "breakout_direction": "bearish",
                    "range_size": range_size,
                },
            )

            if await self.validate_signal(signal):
                self.last_breakout_time = data.timestamp
                logger.info(
                    "Bearish breakout signal generated",
                    strategy=self.name,
                    symbol=data.symbol,
                    level=level,
                    breakout_price=breakout_price,
                    confidence=confidence,
                )
                return signal

            return None

        except Exception as e:
            logger.error(
                "Bearish breakout signal generation failed", strategy=self.name, error=str(e)
            )
            return None

    async def _generate_false_breakout_exit_signal(
        self, data: MarketData, false_breakout_info: dict[str, Any]
    ) -> Signal | None:
        """Generate false breakout exit signal.

        Args:
            data: Market data
            false_breakout_info: False breakout information

        Returns:
            Exit signal or None if validation fails
        """
        try:
            # Determine exit direction based on false breakout type
            if false_breakout_info["breakout_type"] == "false_resistance":
                direction = SignalDirection.SELL
            else:  # false_support
                direction = SignalDirection.BUY

            signal = Signal(
                direction=direction,
                confidence=0.9,  # High confidence for false breakout exits
                timestamp=data.timestamp,
                symbol=data.symbol,
                strategy_name=self.name,
                metadata={
                    "signal_type": "false_breakout_exit",
                    "false_breakout_level": false_breakout_info["level"],
                    "current_price": false_breakout_info["current_price"],
                    "breakout_type": false_breakout_info["breakout_type"],
                },
            )

            if await self.validate_signal(signal):
                logger.info(
                    "False breakout exit signal generated",
                    strategy=self.name,
                    symbol=data.symbol,
                    direction=direction.value,
                    level=false_breakout_info["level"],
                )
                return signal

            return None

        except Exception as e:
            logger.error(
                "False breakout exit signal generation failed", strategy=self.name, error=str(e)
            )
            return None

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate breakout signal before execution.

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
                    min_confidence=self.config.min_confidence,
                )
                return False

            # Check if signal is too old (more than 5 minutes for breakout
            # signals)
            if datetime.now(signal.timestamp.tzinfo) - signal.timestamp > timedelta(minutes=5):
                logger.debug("Signal too old", strategy=self.name, signal_age_minutes=5)
                return False

            # Validate signal metadata
            metadata = signal.metadata
            if "signal_type" not in metadata:
                logger.warning("Missing signal_type in signal metadata", strategy=self.name)
                return False

            # Additional validation for breakout entry signals
            if metadata.get("signal_type") == "breakout_entry":
                required_fields = [
                    "breakout_level",
                    "breakout_price",
                    "target_price",
                    "breakout_direction",
                ]
                for field in required_fields:
                    if field not in metadata:
                        logger.warning(f"Missing {field} in signal metadata", strategy=self.name)
                        return False

                # Validate breakout price
                breakout_price = metadata.get("breakout_price")
                if breakout_price is None or breakout_price <= 0:
                    logger.warning(
                        "Invalid breakout price", strategy=self.name, breakout_price=breakout_price
                    )
                    return False

            return True

        except Exception as e:
            logger.error("Signal validation failed", strategy=self.name, error=str(e))
            return False

    def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size for breakout signal.

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

            # Adjust based on breakout strength
            metadata = signal.metadata
            if "range_size" in metadata:
                range_size = metadata.get("range_size", 0)
                breakout_strength = min(range_size / float(metadata.get("breakout_price", 1)), 1.0)
            else:
                breakout_strength = 1.0

            # Calculate final position size
            position_size = (
                base_size * Decimal(str(confidence_factor)) * Decimal(str(breakout_strength))
            )

            # Ensure position size is within limits
            max_size = Decimal(str(self.config.parameters.get("max_position_size_pct", 0.1)))
            position_size = min(position_size, max_size)

            logger.debug(
                "Position size calculated",
                strategy=self.name,
                base_size=float(base_size),
                confidence_factor=confidence_factor,
                breakout_strength=breakout_strength,
                final_size=float(position_size),
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

            # Check ATR-based stop loss
            if (
                len(self.high_history) >= self.atr_period + 1
            ):  # Need at least period + 1 data points
                atr = calculate_atr(
                    # Pass more data for talib
                    self.high_history[-(self.atr_period + 1) :],
                    self.low_history[-(self.atr_period + 1) :],
                    self.price_history[-(self.atr_period + 1) :],
                )

                if atr is not None:
                    current_price = float(position.current_price)
                    entry_price = float(position.entry_price)

                    # Calculate stop loss distance
                    stop_distance = atr * self.atr_multiplier

                    # Check if price has moved against position beyond stop
                    # loss
                    if position.side.value == "buy":
                        stop_price = entry_price - stop_distance
                        if current_price <= stop_price:
                            logger.info(
                                "ATR stop loss triggered",
                                strategy=self.name,
                                symbol=position.symbol,
                                entry_price=entry_price,
                                current_price=current_price,
                                stop_price=stop_price,
                            )
                            return True
                    else:  # sell position
                        stop_price = entry_price + stop_distance
                        if current_price >= stop_price:
                            logger.info(
                                "ATR stop loss triggered",
                                strategy=self.name,
                                symbol=position.symbol,
                                entry_price=entry_price,
                                current_price=current_price,
                                stop_price=stop_price,
                            )
                            return True

            # Check target price
            if "target_price" in position.metadata:
                target_price = float(position.metadata["target_price"])
                current_price = float(position.current_price)

                if position.side.value == "buy" and current_price >= target_price:
                    logger.info(
                        "Target price reached",
                        strategy=self.name,
                        symbol=position.symbol,
                        target_price=target_price,
                        current_price=current_price,
                    )
                    return True
                elif position.side.value == "sell" and current_price <= target_price:
                    logger.info(
                        "Target price reached",
                        strategy=self.name,
                        symbol=position.symbol,
                        target_price=target_price,
                        current_price=current_price,
                    )
                    return True

            return False

        except Exception as e:
            logger.error("Exit check failed", strategy=self.name, error=str(e))
            return False

    def get_strategy_info(self) -> dict[str, Any]:
        """Get breakout strategy information.

        Returns:
            Strategy information dictionary
        """
        base_info = super().get_strategy_info()

        # Add strategy-specific information
        strategy_info = {
            **base_info,
            "strategy_type": "breakout",
            "parameters": {
                "lookback_period": self.lookback_period,
                "breakout_threshold": self.breakout_threshold,
                "volume_multiplier": self.volume_multiplier,
                "consolidation_periods": self.consolidation_periods,
                "false_breakout_filter": self.false_breakout_filter,
                "false_breakout_threshold": self.false_breakout_threshold,
                "target_multiplier": self.target_multiplier,
                "atr_period": self.atr_period,
                "atr_multiplier": self.atr_multiplier,
            },
            "price_history_length": len(self.price_history),
            "volume_history_length": len(self.volume_history),
            "support_levels_count": len(self.support_levels),
            "resistance_levels_count": len(self.resistance_levels),
            "last_breakout_time": self.last_breakout_time,
        }

        return strategy_info
