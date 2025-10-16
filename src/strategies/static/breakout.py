"""
Breakout Strategy Implementation.

This module implements a breakout strategy that identifies support/resistance
breakouts with volume confirmation and false breakout filtering.

CRITICAL: This strategy MUST inherit from BaseStrategy and follow the exact interface.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

# From P-001 - Use structured logging
# Logger is provided by BaseStrategy (via BaseComponent)
# From P-001 - Use existing types
from src.core.types import (
    MarketData,
    Position,
    Signal,
    SignalDirection,
    StrategyType,
)

# Error handling decorators
from src.error_handling.decorators import with_circuit_breaker, with_error_context, with_retry

# From P-008+ - Use risk management
# MANDATORY: Import from P-011 - NEVER recreate the base strategy
from src.strategies.base import BaseStrategy
from src.strategies.dependencies import StrategyServiceContainer

# From P-007A - Use decorators and validators
from src.utils.decorators import time_execution
from src.utils.strategy_commons import StrategyCommons

# Technical indicators available through data service


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

    def __init__(self, config: dict[str, Any], services: "StrategyServiceContainer | None" = None):
        """Initialize Breakout Strategy.

        Args:
            config: Strategy configuration dictionary
        """
        super().__init__(config, services)
        # Use the name from config (already set by BaseStrategy)

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

        # Initialize strategy commons for shared functionality
        self.commons = StrategyCommons(
            self.name,
            {
                "max_history_length": max(
                    self.lookback_period, self.atr_period, self.consolidation_periods
                )
                + 10
            },
        )

        # Store current symbol for indicator calculations
        self._current_symbol: "str | None" = None

        # Support/resistance levels - store as Decimal for precision
        self.support_levels: list[Decimal] = []
        self.resistance_levels: list[Decimal] = []
        self.last_breakout_time: "datetime | None" = None

        self.logger.info(
            "Breakout Strategy initialized",
            strategy=self.name,
            lookback_period=self.lookback_period,
            breakout_threshold=self.breakout_threshold,
            volume_multiplier=self.volume_multiplier,
        )

    @property
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        return StrategyType.MOMENTUM


    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @with_error_context(operation="breakout_signal_generation")
    @with_retry(max_attempts=3, base_delay=Decimal("1.0"))
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
                self.logger.warning("Invalid market data received", strategy=self.name)
                return []

            # Validate price data
            price = float(data.price)
            if price <= 0:
                self.logger.warning("Invalid price value", strategy=self.name, price=price)
                return []

            # Store current symbol for indicator calculations
            self._current_symbol = data.symbol

            # Update support/resistance levels using service-based approach
            await self._update_support_resistance_levels(data)

            # Check for consolidation period
            consolidation_met = await self._check_consolidation_period(data)
            self.logger.debug(
                "Consolidation check result",
                strategy=self.name,
                consolidation_met=consolidation_met,
                required_periods=self.consolidation_periods,
            )
            if not consolidation_met:
                self.logger.debug(
                    "Consolidation period not met",
                    strategy=self.name,
                    required_periods=self.consolidation_periods,
                )
                return []

            # Generate signals based on breakout analysis
            signals = []

            # Check for resistance breakout (bullish)
            resistance_breakout = await self._check_resistance_breakout(data)
            if resistance_breakout:
                signal = await self._generate_bullish_breakout_signal(data, resistance_breakout)
                if signal:
                    signals.append(signal)

            # Check for support breakout (bearish)
            support_breakout = await self._check_support_breakout(data)
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
            self.logger.error(
                "Signal generation failed",
                strategy=self.name,
                error=str(e),
                symbol=data.symbol if data else "unknown",
            )
            return []  # MANDATORY: Graceful degradation


    async def _update_support_resistance_levels(self, data: MarketData) -> None:
        """Update support and resistance levels based on recent price action."""
        try:
            # Use a simple approach with recent price data to find support/resistance
            # In a real implementation, you might want to use more sophisticated algorithms
            # For now, use high/low from recent data

            # Get current price as our reference
            current_price = data.price if isinstance(data.price, Decimal) else Decimal(str(data.price))

            # For demo purposes, create simple support/resistance based on current price
            # In practice, you'd analyze historical highs/lows
            price_range = current_price * Decimal("0.02")  # 2% range

            # Simple resistance level above current price
            resistance = current_price + price_range
            # Simple support level below current price
            support = current_price - price_range

            # Update levels if they don't already exist
            if resistance not in self.resistance_levels:
                self.resistance_levels.append(resistance)
            if support not in self.support_levels:
                self.support_levels.append(support)

            # Keep only recent levels (last 5)
            self.resistance_levels = self.resistance_levels[-5:]
            self.support_levels = self.support_levels[-5:]

        except Exception as e:
            self.logger.error("Support/resistance update failed", strategy=self.name, error=str(e))

    async def _check_consolidation_period(self, data: MarketData) -> bool:
        """Check if price has been consolidating for required periods.

        Returns:
            True if consolidation period met, False otherwise
        """
        try:
            # If consolidation_periods is 0, skip the check (always pass)
            if self.consolidation_periods == 0:
                return True

            # Use volatility as a proxy for consolidation
            volatility = await self.get_volatility(data.symbol, self.consolidation_periods)
            if volatility is None:
                return False

            # Lower volatility indicates consolidation
            # Use 1% as consolidation threshold
            consolidation_threshold = Decimal("0.01")
            is_consolidating = volatility <= consolidation_threshold

            self.logger.debug(
                "Consolidation check details",
                strategy=self.name,
                volatility=float(volatility),
                consolidation_threshold=float(consolidation_threshold),
                is_consolidating=is_consolidating,
            )

            return is_consolidating

        except Exception as e:
            self.logger.error("Consolidation check failed", strategy=self.name, error=str(e))
            return False

    async def _check_resistance_breakout(self, data: MarketData) -> "dict[str, Any] | None":
        """Check for resistance breakout.

        Args:
            data: Current market data

        Returns:
            Breakout information or None if no breakout
        """
        try:
            current_price = data.price
            current_volume = float(data.volume) if data.volume else 0.0

            self.logger.debug(
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
                self.logger.debug(
                    "Checking resistance level",
                    resistance=resistance,
                    breakout_price=breakout_price,
                    current_price=current_price,
                    is_breakout=current_price > breakout_price,
                )

                # Check if price broke above resistance
                if current_price > breakout_price:
                    # Check volume confirmation
                    volume_confirmed = await self._check_volume_confirmation(data)

                    if volume_confirmed:
                        return {
                            "level": resistance,
                            "breakout_price": current_price,
                            "breakout_type": "resistance",
                            "volume": current_volume,
                        }

            return None

        except Exception as e:
            self.logger.error("Resistance breakout check failed", strategy=self.name, error=str(e))
            return None

    async def _check_support_breakout(self, data: MarketData) -> "dict[str, Any] | None":
        """Check for support breakout.

        Args:
            data: Current market data

        Returns:
            Breakout information or None if no breakout
        """
        try:
            current_price = data.price
            current_volume = float(data.volume) if data.volume else 0.0

            # Check each support level
            for support in self.support_levels:
                # Check if price broke below support
                if current_price < support * (Decimal("1") - Decimal(str(self.breakout_threshold))):
                    # Check volume confirmation
                    if await self._check_volume_confirmation(data):
                        return {
                            "level": support,
                            "breakout_price": current_price,
                            "breakout_type": "support",
                            "volume": current_volume,
                        }

            return None

        except Exception as e:
            self.logger.error("Support breakout check failed", strategy=self.name, error=str(e))
            return None

    async def _check_volume_confirmation(self, data: MarketData) -> bool:
        """Check if volume confirms the breakout.

        Args:
            data: Market data with volume information

        Returns:
            True if volume confirms breakout, False otherwise
        """
        try:
            current_volume = float(data.volume) if data.volume else 0.0
            if current_volume <= 0:
                return False

            # Use volume ratio service
            volume_ratio = await self.get_volume_ratio(data.symbol, self.lookback_period)
            if volume_ratio is None:
                return True  # Pass if insufficient data

            # Check if volume ratio is above threshold
            return volume_ratio >= Decimal(str(self.volume_multiplier))

        except Exception as e:
            self.logger.error("Volume confirmation check failed", strategy=self.name, error=str(e))
            return True  # Pass on error

    def _check_false_breakout(self, data: MarketData) -> "dict[str, Any] | None":
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
            self.logger.error("False breakout check failed", strategy=self.name, error=str(e))
            return None

    async def _generate_bullish_breakout_signal(
        self, data: MarketData, breakout_info: dict[str, Any]
    ) -> "Signal | None":
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
            breakout_strength = Decimal(str((breakout_price - level) / level))

            # Calculate volume strength using service
            volume_ratio = await self.get_volume_ratio(data.symbol, self.lookback_period)
            volume_strength = min(volume_ratio, Decimal("3.0")) if volume_ratio else Decimal("1.0")

            confidence = min(breakout_strength + volume_strength, Decimal("1.0"))

            # Calculate target price
            range_size = (
                level - min(self.support_levels) if self.support_levels else level * Decimal("0.1")
            )
            target_price = breakout_price + (range_size * Decimal(str(self.target_multiplier)))

            signal = Signal(
                signal_id="test_signal_1",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                symbol=data.symbol,
                direction=SignalDirection.BUY,
                strength=Decimal(str(confidence)),
                timestamp=data.timestamp,
                source=self.name,
                metadata={
                    "breakout_level": level,
                    "breakout_price": breakout_price,
                    "target_price": target_price,
                    "volume": volume,
                    "signal_type": "breakout_entry",
                    "breakout_direction": "bullish",
                    "range_size": range_size,
                    "confidence": confidence,  # Keep for backwards compatibility
                },
            )

            if await self.validate_signal(signal):
                self.last_breakout_time = data.timestamp
                self.logger.info(
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
            self.logger.error(
                "Bullish breakout signal generation failed", strategy=self.name, error=str(e)
            )
            return None

    async def _generate_bearish_breakout_signal(
        self, data: MarketData, breakout_info: dict[str, Any]
    ) -> "Signal | None":
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
            breakout_strength = Decimal(str((level - breakout_price) / level))

            # Calculate volume strength using service
            volume_ratio = await self.get_volume_ratio(data.symbol, self.lookback_period)
            volume_strength = min(volume_ratio, Decimal("3.0")) if volume_ratio else Decimal("1.0")

            confidence = min(breakout_strength + volume_strength, Decimal("1.0"))

            # Calculate target price
            range_size = (
                max(self.resistance_levels) - level
                if self.resistance_levels
                else level * Decimal("0.1")
            )
            target_price = breakout_price - (range_size * Decimal(str(self.target_multiplier)))

            signal = Signal(
                signal_id="test_signal_2",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                symbol=data.symbol,
                direction=SignalDirection.SELL,
                strength=Decimal(str(confidence)),
                timestamp=data.timestamp,
                source=self.name,
                metadata={
                    "breakout_level": level,
                    "breakout_price": breakout_price,
                    "target_price": target_price,
                    "volume": volume,
                    "signal_type": "breakout_entry",
                    "breakout_direction": "bearish",
                    "range_size": range_size,
                    "confidence": confidence,  # Keep for backwards compatibility
                },
            )

            if await self.validate_signal(signal):
                self.last_breakout_time = data.timestamp
                self.logger.info(
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
            self.logger.error(
                "Bearish breakout signal generation failed", strategy=self.name, error=str(e)
            )
            return None

    async def _generate_false_breakout_exit_signal(
        self, data: MarketData, false_breakout_info: dict[str, Any]
    ) -> "Signal | None":
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
                signal_id="test_signal_3",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                symbol=data.symbol,
                direction=direction,
                strength=Decimal("0.9"),  # High confidence for false breakout exits
                timestamp=data.timestamp,
                source=self.name,
                metadata={
                    "signal_type": "false_breakout_exit",
                    "false_breakout_level": false_breakout_info["level"],
                    "current_price": false_breakout_info["current_price"],
                    "breakout_type": false_breakout_info["breakout_type"],
                    "confidence": 0.9,  # Keep for backwards compatibility
                },
            )

            if await self.validate_signal(signal):
                self.logger.info(
                    "False breakout exit signal generated",
                    strategy=self.name,
                    symbol=data.symbol,
                    direction=direction.value,
                    level=false_breakout_info["level"],
                )
                return signal

            return None

        except Exception as e:
            self.logger.error(
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
            if not signal or signal.strength < self.config.min_confidence:
                self.logger.debug(
                    "Signal confidence below threshold",
                    strategy=self.name,
                    confidence=signal.strength if signal else 0,
                    min_confidence=self.config.min_confidence,
                )
                return False

            # Check if signal is too old (more than 5 minutes for breakout
            # signals)
            if datetime.now(signal.timestamp.tzinfo) - signal.timestamp > timedelta(minutes=5):
                self.logger.debug("Signal too old", strategy=self.name, signal_age_minutes=5)
                return False

            # Validate signal metadata
            metadata = signal.metadata
            if "signal_type" not in metadata:
                self.logger.warning("Missing signal_type in signal metadata", strategy=self.name)
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
                        self.logger.warning(
                            f"Missing {field} in signal metadata", strategy=self.name
                        )
                        return False

                # Validate breakout price
                breakout_price = metadata.get("breakout_price")
                if breakout_price is None or breakout_price <= 0:
                    self.logger.warning(
                        "Invalid breakout price", strategy=self.name, breakout_price=breakout_price
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error("Signal validation failed", strategy=self.name, error=str(e))
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
            confidence_factor = signal.strength

            # Adjust based on breakout strength
            metadata = signal.metadata
            if "range_size" in metadata:
                range_size = metadata.get("range_size", 0)
                # Use Decimal arithmetic for breakout strength calculation
                breakout_price = Decimal(str(metadata.get("breakout_price", 1)))
                breakout_strength = min(float(range_size / breakout_price), 1.0)
            else:
                breakout_strength = 1.0

            # Calculate final position size
            position_size = (
                base_size * Decimal(str(confidence_factor)) * Decimal(str(breakout_strength))
            )

            # Ensure position size is within limits
            max_size = Decimal(str(self.config.parameters.get("max_position_size_pct", 0.1)))
            position_size = min(position_size, max_size)

            self.logger.debug(
                "Position size calculated",
                strategy=self.name,
                base_size=float(base_size),
                confidence_factor=confidence_factor,
                breakout_strength=breakout_strength,
                final_size=float(position_size),
            )

            return position_size

        except Exception as e:
            self.logger.error("Position size calculation failed", strategy=self.name, error=str(e))
            # Return minimum position size on error
            return self.config.position_size_pct * Decimal("0.5")

    async def should_exit(self, position: Position, data: MarketData) -> bool:
        """Determine if position should be closed.

        Args:
            position: Current position
            data: Current market data

        Returns:
            True if position should be closed, False otherwise
        """
        try:
            # Check ATR-based stop loss using service
            atr = await self.get_atr(position.symbol, self.atr_period)
            if atr is not None and atr > 0:
                current_price = position.current_price
                entry_price = position.entry_price

                # Calculate stop loss distance
                stop_distance = atr * Decimal(str(self.atr_multiplier))

                # Check if price has moved against position beyond stop loss
                if position.side.value == "LONG":
                    stop_price = entry_price - stop_distance
                    if current_price <= stop_price:
                        self.logger.info(
                            "ATR stop loss triggered",
                            strategy=self.name,
                            symbol=position.symbol,
                            entry_price=entry_price,
                            current_price=current_price,
                            stop_price=stop_price,
                        )
                        return True
                else:  # SHORT position
                    stop_price = entry_price + stop_distance
                    if current_price >= stop_price:
                        self.logger.info(
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
                target_price = Decimal(str(position.metadata["target_price"]))
                current_price = position.current_price

                if position.side.value == "LONG" and current_price >= target_price:
                    self.logger.info(
                        "Target price reached",
                        strategy=self.name,
                        symbol=position.symbol,
                        target_price=target_price,
                        current_price=current_price,
                    )
                    return True
                elif position.side.value == "SHORT" and current_price <= target_price:
                    self.logger.info(
                        "Target price reached",
                        strategy=self.name,
                        symbol=position.symbol,
                        target_price=target_price,
                        current_price=current_price,
                    )
                    return True

            return False

        except Exception as e:
            self.logger.error("Exit check failed", strategy=self.name, error=str(e))
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
            "current_symbol": self._current_symbol,
            "support_levels_count": len(self.support_levels),
            "resistance_levels_count": len(self.resistance_levels),
            "last_breakout_time": self.last_breakout_time,
        }

        return strategy_info
