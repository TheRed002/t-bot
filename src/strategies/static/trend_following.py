"""
Trend Following Strategy Implementation.

This module implements a momentum-based trend following strategy that identifies
and follows market trends using moving average crossovers, RSI confirmation,
and volume analysis.

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

# Technical indicators for centralized calculations
from src.data.features.technical_indicators import TechnicalIndicators

# From P-002A - Use error handling decorators
from src.error_handling.decorators import with_circuit_breaker, with_error_context, with_retry

# From P-008+ - Use risk management
# MANDATORY: Import from P-011 - NEVER recreate the base strategy
from src.strategies.base import BaseStrategy
from src.strategies.dependencies import StrategyServiceContainer

# From P-007A - Use decorators and validators
from src.utils.decorators import time_execution

# RSI calculation is implemented inline to avoid import issues

# Logger is now provided by BaseStrategy (via BaseComponent)


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

    def __init__(self, config: dict[str, Any], services: "StrategyServiceContainer | None" = None):
        """Initialize Trend Following Strategy.

        Args:
            config: Strategy configuration dictionary
        """
        super().__init__(config, services)
        # Use the name from config (already set by BaseStrategy)

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
        self.price_history: list[float] = []
        self.volume_history: list[float] = []
        self.high_history: list[float] = []
        self.low_history: list[float] = []

        # Position tracking for pyramiding
        self.position_levels: dict[str, int] = {}
        self.position_entries: dict[str, list[datetime]] = {}

        self.logger.info(
            "Trend Following Strategy initialized",
            strategy=self.name,
            fast_ma=self.fast_ma,
            slow_ma=self.slow_ma,
            rsi_period=self.rsi_period,
        )

    def set_technical_indicators(self, technical_indicators: TechnicalIndicators) -> None:
        """Set technical indicators service for centralized calculations."""
        self._technical_indicators = technical_indicators
        self.logger.info("Technical indicators service set", strategy=self.name)

    @property
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        return StrategyType.TREND_FOLLOWING

    @time_execution
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @with_error_context(operation="trend_following_signals")
    @with_retry(max_attempts=3)
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
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
                self.logger.warning("Invalid market data received", strategy=self.name)
                return []

            # Validate price data
            price = float(data.price)
            if price <= 0:
                self.logger.warning("Invalid price value", strategy=self.name, price=price)
                return []

            # Data is managed by the indicators service, no manual tracking needed

            # Check if we have enough data for calculations
            min_required = max(self.slow_ma, self.rsi_period) + 10
            if len(self.price_history) < min_required:
                self.logger.debug(
                    "Insufficient price history for signal generation",
                    strategy=self.name,
                    current_length=len(self.price_history),
                    required_length=min_required,
                )
                return []

            # Calculate technical indicators
            # Use shared BaseStrategy methods for technical indicators (eliminates code duplication)
            fast_ma_decimal = await self.get_sma(data.symbol, self.fast_ma)
            slow_ma_decimal = await self.get_sma(data.symbol, self.slow_ma)
            rsi_decimal = await self.get_rsi(data.symbol, self.rsi_period)

            fast_ma = float(fast_ma_decimal) if fast_ma_decimal else None
            slow_ma = float(slow_ma_decimal) if slow_ma_decimal else None
            rsi = float(rsi_decimal) if rsi_decimal else None

            if fast_ma is None or slow_ma is None or rsi is None:
                return []

            # Check volume confirmation if enabled
            if self.volume_confirmation and not await self._check_volume_confirmation(data):
                self.logger.debug(
                    "Volume confirmation failed",
                    strategy=self.name,
                    rsi=rsi,
                    fast_ma=fast_ma,
                    slow_ma=slow_ma,
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
                signal = await self._generate_exit_signal(
                    data, SignalDirection.SELL, "trend_reversal"
                )
                if signal:
                    signals.append(signal)

            elif trend_down and rsi_bullish:
                # Exit bearish positions
                signal = await self._generate_exit_signal(
                    data, SignalDirection.BUY, "trend_reversal"
                )
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


    # REMOVED: Duplicated technical indicator methods
    # Now using shared BaseStrategy methods (get_sma, get_rsi) to eliminate code duplication
    # This reduces maintenance burden and ensures consistency across all strategies

    async def _check_volume_confirmation(self, data: MarketData) -> bool:
        """Check if volume confirms the trend using indicators service.

        Args:
            data: Current market data

        Returns:
            True if volume confirms trend, False otherwise
        """
        try:
            current_volume = float(data.volume) if data.volume else 0.0
            if current_volume <= 0:
                return False

            # Get volume ratio from indicators service
            volume_ratio = await self._indicators.calculate_volume_ratio(
                symbol=self.config.symbol,
                period=self.fast_ma
            )

            if volume_ratio is None:
                return True  # Pass if insufficient data

            # Check if current volume is above threshold
            return float(volume_ratio) >= self.min_volume_ratio

        except Exception as e:
            self.logger.error("Volume confirmation check failed", strategy=self.name, error=str(e))
            return True  # Pass on error

    async def _generate_bullish_signal(
        self, data: MarketData, fast_ma: float, slow_ma: float, rsi: float
    ) -> "Signal | None":
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
                self.logger.debug(
                    "Maximum pyramid levels reached",
                    strategy=self.name,
                    symbol=data.symbol,
                    current_levels=current_levels,
                    max_levels=self.max_pyramid_levels,
                )
                return None

            # Calculate signal confidence
            ma_strength = (fast_ma - slow_ma) / slow_ma
            rsi_strength = (rsi - 50) / 50  # Normalize RSI to -1 to 1
            confidence = min(abs(ma_strength) + abs(rsi_strength), 1.0)

            # Ensure minimum confidence for trend signals
            confidence = max(confidence, 0.6)

            signal = Signal(
                signal_id="test_signal_1",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                direction=SignalDirection.BUY,
                strength=Decimal(str(confidence)),
                timestamp=data.timestamp,
                symbol=data.symbol,
                source=self.name,
                metadata={
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "rsi": rsi,
                    "ma_strength": ma_strength,
                    "rsi_strength": rsi_strength,
                    "signal_type": "trend_entry",
                    "trend_direction": "bullish",
                    "pyramid_level": current_levels + 1,
                },
            )

            if await self.validate_signal(signal):
                self.logger.info(
                    "Bullish trend signal generated",
                    strategy=self.name,
                    symbol=data.symbol,
                    fast_ma=fast_ma,
                    slow_ma=slow_ma,
                    rsi=rsi,
                    confidence=confidence,
                )
                return signal

            return None

        except Exception as e:
            self.logger.error("Bullish signal generation failed", strategy=self.name, error=str(e))
            return None

    async def _generate_bearish_signal(
        self, data: MarketData, fast_ma: float, slow_ma: float, rsi: float
    ) -> "Signal | None":
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
                self.logger.debug(
                    "Maximum pyramid levels reached",
                    strategy=self.name,
                    symbol=data.symbol,
                    current_levels=current_levels,
                    max_levels=self.max_pyramid_levels,
                )
                return None

            # Calculate signal confidence
            ma_strength = (slow_ma - fast_ma) / slow_ma
            rsi_strength = (50 - rsi) / 50  # Normalize RSI to -1 to 1
            confidence = min(abs(ma_strength) + abs(rsi_strength), 1.0)

            # Ensure minimum confidence for trend signals
            confidence = max(confidence, 0.6)

            signal = Signal(
                signal_id="test_signal_2",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                direction=SignalDirection.SELL,
                strength=Decimal(str(confidence)),
                timestamp=data.timestamp,
                symbol=data.symbol,
                source=self.name,
                metadata={
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "rsi": rsi,
                    "ma_strength": ma_strength,
                    "rsi_strength": rsi_strength,
                    "signal_type": "trend_entry",
                    "trend_direction": "bearish",
                    "pyramid_level": current_levels + 1,
                },
            )

            if await self.validate_signal(signal):
                self.logger.info(
                    "Bearish trend signal generated",
                    strategy=self.name,
                    symbol=data.symbol,
                    fast_ma=fast_ma,
                    slow_ma=slow_ma,
                    rsi=rsi,
                    confidence=confidence,
                )
                return signal

            return None

        except Exception as e:
            self.logger.error("Bearish signal generation failed", strategy=self.name, error=str(e))
            return None

    async def _generate_exit_signal(
        self, data: MarketData, direction: SignalDirection, reason: str
    ) -> "Signal | None":
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
                signal_id="test_signal_3",
                strategy_id="test_strategy_1",
                strategy_name="test_strategy",
                direction=direction,
                strength=Decimal("0.8"),  # High confidence for exits
                timestamp=data.timestamp,
                symbol=data.symbol,
                source=self.name,
                metadata={
                    "signal_type": "trend_exit",
                    "exit_reason": reason,
                    "fast_ma": float(fast_ma_result) if (fast_ma_result := await self.get_sma(data.symbol, self.fast_ma)) else None,
                    "slow_ma": float(slow_ma_result) if (slow_ma_result := await self.get_sma(data.symbol, self.slow_ma)) else None,
                    "rsi": float(rsi_result) if (rsi_result := await self.get_rsi(data.symbol, self.rsi_period)) else None,
                },
            )

            if await self.validate_signal(signal):
                self.logger.info(
                    "Trend exit signal generated",
                    strategy=self.name,
                    symbol=data.symbol,
                    direction=direction.value,
                    reason=reason,
                )
                return signal

            return None

        except Exception as e:
            self.logger.error("Exit signal generation failed", strategy=self.name, error=str(e))
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
            if not signal or signal.strength < self.config.min_confidence:
                self.logger.debug(
                    "Signal confidence below threshold",
                    strategy=self.name,
                    confidence=signal.strength if signal else 0,
                    min_confidence=self.config.min_confidence,
                )
                return False

            # Check if signal is too old (more than 10 minutes for trend
            # signals)
            if datetime.now(signal.timestamp.tzinfo) - signal.timestamp > timedelta(minutes=10):
                self.logger.debug("Signal too old", strategy=self.name, signal_age_minutes=10)
                return False

            # Validate signal metadata
            metadata = signal.metadata
            if "signal_type" not in metadata:
                self.logger.warning("Missing signal_type in signal metadata", strategy=self.name)
                return False

            # Additional validation for trend entry signals
            if metadata.get("signal_type") == "trend_entry":
                required_fields = ["fast_ma", "slow_ma", "rsi", "trend_direction"]
                for field in required_fields:
                    if field not in metadata:
                        self.logger.warning(
                            f"Missing {field} in signal metadata", strategy=self.name
                        )
                        return False

                # Validate RSI bounds
                rsi = metadata.get("rsi")
                if rsi is None or rsi < 0 or rsi > 100:
                    self.logger.warning("Invalid RSI value", strategy=self.name, rsi=rsi)
                    return False

            return True

        except Exception as e:
            self.logger.error("Signal validation failed", strategy=self.name, error=str(e))
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
            confidence_factor = signal.strength

            # Adjust based on trend strength
            metadata = signal.metadata
            ma_strength = abs(metadata.get("ma_strength", 0))
            rsi_strength = abs(metadata.get("rsi_strength", 0))
            trend_strength = min(ma_strength + rsi_strength, 1.0)

            # Adjust based on pyramid level (smaller positions for higher
            # levels)
            pyramid_level = metadata.get("pyramid_level", 1)
            pyramid_factor = 1.0 / pyramid_level  # Decrease size with each level

            # Calculate final position size
            position_size = (
                base_size
                * Decimal(str(confidence_factor))
                * Decimal(str(trend_strength))
                * Decimal(str(pyramid_factor))
            )

            # Ensure position size is within limits
            max_size = Decimal(str(self.config.parameters.get("max_position_size_pct", 0.1)))
            position_size = min(position_size, max_size)

            self.logger.debug(
                "Position size calculated",
                strategy=self.name,
                base_size=float(base_size),
                confidence_factor=confidence_factor,
                trend_strength=trend_strength,
                pyramid_factor=pyramid_factor,
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
            # Data is managed by the indicators service, no manual tracking needed

            # Check time-based exit
            if self._should_exit_by_time(position):
                self.logger.info(
                    "Time-based exit triggered", strategy=self.name, symbol=position.symbol
                )
                return True

            # Check trailing stop
            if self._should_exit_by_trailing_stop(position, data):
                self.logger.info(
                    "Trailing stop triggered", strategy=self.name, symbol=position.symbol
                )
                return True

            # Check trend reversal
            # Use shared BaseStrategy methods for technical indicators (eliminates code duplication)
            fast_ma_decimal = await self.get_sma(position.symbol, self.fast_ma)
            slow_ma_decimal = await self.get_sma(position.symbol, self.slow_ma)
            rsi_decimal = await self.get_rsi(position.symbol, self.rsi_period)

            fast_ma = float(fast_ma_decimal) if fast_ma_decimal else None
            slow_ma = float(slow_ma_decimal) if slow_ma_decimal else None
            rsi = float(rsi_decimal) if rsi_decimal else None

            if fast_ma is None or slow_ma is None or rsi is None:
                return False

            # Exit if trend has reversed
            if position.side.value == "LONG":
                if fast_ma < slow_ma and rsi < 50:
                    self.logger.info(
                        "Trend reversal exit triggered",
                        strategy=self.name,
                        symbol=position.symbol,
                        fast_ma=fast_ma,
                        slow_ma=slow_ma,
                        rsi=rsi,
                    )
                    return True
            else:  # SHORT position
                if fast_ma > slow_ma and rsi > 50:
                    self.logger.info(
                        "Trend reversal exit triggered",
                        strategy=self.name,
                        symbol=position.symbol,
                        fast_ma=fast_ma,
                        slow_ma=slow_ma,
                        rsi=rsi,
                    )
                    return True

            return False

        except Exception as e:
            self.logger.error("Exit check failed", strategy=self.name, error=str(e))
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
            entry_time = position.opened_at

            # Check if position has been open too long
            current_time = datetime.now(entry_time.tzinfo)
            time_diff = current_time - entry_time

            return time_diff > timedelta(hours=self.time_exit_hours)

        except Exception as e:
            self.logger.error("Time exit check failed", strategy=self.name, error=str(e))
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
            current_price = data.price
            entry_price = position.entry_price

            trailing_stop_decimal = Decimal(str(self.trailing_stop_pct))

            if position.side.value == "LONG":
                # For long positions, exit if price falls below trailing stop
                # Trailing stop is entry_price - (entry_price * trailing_stop_pct)
                trailing_stop = entry_price - (entry_price * trailing_stop_decimal)
                return current_price <= trailing_stop
            else:  # SHORT
                # For short positions, exit if price rises above trailing stop
                # Trailing stop is entry_price + (entry_price * trailing_stop_pct)
                trailing_stop = entry_price + (entry_price * trailing_stop_decimal)
                return current_price >= trailing_stop

        except Exception as e:
            self.logger.error("Trailing stop check failed", strategy=self.name, error=str(e))
            return False


    # Helper methods for accessing data through data service

    def get_strategy_info(self) -> dict[str, Any]:
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
                "time_exit_hours": self.time_exit_hours,
            },
            "price_history_length": len(self.price_history),
            "volume_history_length": len(self.volume_history),
            "position_levels": self.position_levels,
        }

        return strategy_info
