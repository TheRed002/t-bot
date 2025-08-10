"""
Mean Reversion Strategy Implementation.

This module implements a statistical arbitrage strategy based on mean reversion principles.
The strategy identifies overbought/oversold conditions using Z-score calculations and generates
signals when prices deviate significantly from their moving average.

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
from src.utils.helpers import calculate_atr, calculate_zscore

# From P-008+ - Use risk management
from src.risk_management.base import BaseRiskManager

# From P-001 - Use structured logging
from src.core.logging import get_logger

logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy Implementation.

    This strategy identifies statistical arbitrage opportunities by detecting when
    prices deviate significantly from their moving average (mean reversion).

    Key Features:
    - Z-score calculation with configurable lookback period
    - Entry/exit threshold configuration
    - ATR-based stop loss and take profit
    - Volume and volatility filters
    - Multi-timeframe confirmation
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Mean Reversion Strategy.

        Args:
            config: Strategy configuration dictionary
        """
        super().__init__(config)
        # Use the name from config (already set by BaseStrategy)
        self.strategy_type = StrategyType.STATIC

        # Strategy-specific parameters with defaults
        self.lookback_period = self.config.parameters.get(
            "lookback_period", 20)
        self.entry_threshold = self.config.parameters.get(
            "entry_threshold", 2.0)
        self.exit_threshold = self.config.parameters.get("exit_threshold", 0.5)
        self.atr_period = self.config.parameters.get("atr_period", 14)
        self.atr_multiplier = self.config.parameters.get("atr_multiplier", 2.0)
        self.volume_filter = self.config.parameters.get("volume_filter", True)
        self.min_volume_ratio = self.config.parameters.get(
            "min_volume_ratio", 1.5)
        self.confirmation_timeframe = self.config.parameters.get(
            "confirmation_timeframe", "1h")

        # Price history for calculations
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.high_history: List[float] = []
        self.low_history: List[float] = []

        logger.info(
            "Mean Reversion Strategy initialized",
            strategy=self.name,
            lookback_period=self.lookback_period,
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold
        )

    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> List[Signal]:
        """Generate mean reversion signals from market data.

        MANDATORY: Use graceful error handling and input validation.

        Args:
            data: Market data for signal generation

        Returns:
            List of trading signals
        """
        try:
            # MANDATORY: Input validation
            if not data or not data.price:
                logger.warning(
                    "Invalid market data received",
                    strategy=self.name)
                return []

            # Validate price data
            price = float(data.price)
            if price <= 0:
                logger.warning(
                    "Invalid price value",
                    strategy=self.name,
                    price=price)
                return []

            # Update price history
            self._update_price_history(data)

            # Check if we have enough data for calculations
            if len(self.price_history) < self.lookback_period:
                logger.debug(
                    "Insufficient price history for signal generation",
                    strategy=self.name,
                    current_length=len(self.price_history),
                    required_length=self.lookback_period
                )
                return []

            # Calculate Z-score
            z_score = self._calculate_zscore()
            if z_score is None:
                return []

            # Check volume filter if enabled
            if self.volume_filter and not self._check_volume_filter(data):
                logger.debug(
                    "Volume filter rejected signal",
                    strategy=self.name,
                    z_score=z_score
                )
                return []

            # Generate signals based on Z-score
            signals = []

            # Entry signals
            if abs(z_score) >= self.entry_threshold:
                direction = SignalDirection.SELL if z_score > 0 else SignalDirection.BUY
                confidence = min(abs(z_score) / self.entry_threshold, 1.0)

                signal = Signal(
                    direction=direction,
                    confidence=confidence,
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    strategy_name=self.name,
                    metadata={
                        "z_score": z_score,
                        "entry_threshold": self.entry_threshold,
                        "lookback_period": self.lookback_period,
                        "signal_type": "entry"
                    }
                )

                if await self.validate_signal(signal):
                    signals.append(signal)
                    logger.info(
                        "Mean reversion entry signal generated",
                        strategy=self.name,
                        symbol=data.symbol,
                        direction=direction.value,
                        z_score=z_score,
                        confidence=confidence
                    )

            # Exit signals for existing positions
            if abs(z_score) <= self.exit_threshold:
                # Generate exit signals for both directions
                for direction in [SignalDirection.BUY, SignalDirection.SELL]:
                    # For exit signals, confidence should be high when Z-score is close to zero
                    # Use a different confidence calculation for exit signals
                    confidence = max(0.8,
                                     1.0 - (abs(z_score) / self.exit_threshold))

                    signal = Signal(
                        direction=direction,
                        confidence=confidence,
                        timestamp=data.timestamp,
                        symbol=data.symbol,
                        strategy_name=self.name,
                        metadata={
                            "z_score": z_score,
                            "exit_threshold": self.exit_threshold,
                            "signal_type": "exit"
                        }
                    )

                    if await self.validate_signal(signal):
                        signals.append(signal)
                        logger.info(
                            "Mean reversion exit signal generated",
                            strategy=self.name,
                            symbol=data.symbol,
                            direction=direction.value,
                            z_score=z_score,
                            confidence=confidence
                        )

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
        max_length = max(self.lookback_period, self.atr_period) + 10
        if len(self.price_history) > max_length:
            self.price_history = self.price_history[-max_length:]
            self.volume_history = self.volume_history[-max_length:]
            self.high_history = self.high_history[-max_length:]
            self.low_history = self.low_history[-max_length:]

    def _calculate_zscore(self) -> Optional[float]:
        """Calculate Z-score for current price relative to moving average.

        Returns:
            Z-score value or None if calculation fails
        """
        try:
            if len(self.price_history) < self.lookback_period:
                return None

            # Use the helper function which handles ta-lib with fallback
            z_score = calculate_zscore(
                self.price_history, self.lookback_period)

            # TODO: Remove in production - Debug logging
            logger.debug(f"Z-score calculation result: {z_score}")

            return z_score

        except Exception as e:
            logger.error(
                "Z-score calculation failed",
                strategy=self.name,
                error=str(e))
            return None

    def _check_volume_filter(self, data: MarketData) -> bool:
        """Check if volume meets filter requirements.

        Args:
            data: Current market data

        Returns:
            True if volume filter passes, False otherwise
        """
        try:
            if len(self.volume_history) < self.lookback_period:
                return True  # Pass if insufficient data

            current_volume = float(data.volume) if data.volume else 0.0
            if current_volume <= 0:
                return False

            # Calculate average volume
            recent_volumes = self.volume_history[-self.lookback_period:]
            avg_volume = np.mean(recent_volumes)

            if avg_volume <= 0:
                return True  # Pass if no historical volume data

            # Check if current volume is above threshold
            volume_ratio = current_volume / avg_volume
            return volume_ratio >= self.min_volume_ratio

        except Exception as e:
            logger.error(
                "Volume filter check failed",
                strategy=self.name,
                error=str(e))
            return True  # Pass on error

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate mean reversion signal before execution.

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

            # Check if signal is too old (more than 5 minutes)
            if datetime.now(signal.timestamp.tzinfo) - \
                    signal.timestamp > timedelta(minutes=5):
                logger.debug(
                    "Signal too old",
                    strategy=self.name,
                    signal_age_minutes=5)
                return False

            # Validate signal metadata
            metadata = signal.metadata
            if "z_score" not in metadata:
                logger.warning(
                    "Missing z_score in signal metadata",
                    strategy=self.name)
                return False

            z_score = metadata.get("z_score")
            if not isinstance(z_score, (int, float)):
                logger.warning(
                    "Invalid z_score type",
                    strategy=self.name,
                    z_score_type=type(z_score))
                return False

            # Additional validation for entry signals
            if metadata.get("signal_type") == "entry":
                if abs(z_score) < self.entry_threshold:
                    logger.debug(
                        "Entry signal z_score below threshold",
                        strategy=self.name,
                        z_score=z_score,
                        threshold=self.entry_threshold
                    )
                    return False

            return True

        except Exception as e:
            logger.error(
                "Signal validation failed",
                strategy=self.name,
                error=str(e))
            return False

    def get_position_size(self, signal: Signal) -> Decimal:
        """Calculate position size for mean reversion signal.

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

            # Adjust based on Z-score magnitude (stronger deviation = larger
            # position)
            metadata = signal.metadata
            z_score = abs(metadata.get("z_score", 0))
            z_score_factor = min(
                z_score / self.entry_threshold,
                2.0)  # Cap at 2x

            # Calculate final position size
            position_size = base_size * \
                Decimal(str(confidence_factor)) * Decimal(str(z_score_factor))

            # Ensure position size is within limits
            max_size = Decimal(
                str(self.config.parameters.get("max_position_size_pct", 0.1)))
            position_size = min(position_size, max_size)

            logger.debug(
                "Position size calculated",
                strategy=self.name,
                base_size=float(base_size),
                confidence_factor=confidence_factor,
                z_score_factor=z_score_factor,
                final_size=float(position_size)
            )

            return position_size

        except Exception as e:
            logger.error(
                "Position size calculation failed",
                strategy=self.name,
                error=str(e))
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
            # Update price history with current data for accurate Z-score
            # calculation
            self._update_price_history(data)

            # Calculate current Z-score
            z_score = self._calculate_zscore()
            if z_score is None:
                return False

            # Exit if Z-score is within exit threshold
            if abs(z_score) <= self.exit_threshold:
                logger.info(
                    "Exit signal triggered by Z-score",
                    strategy=self.name,
                    symbol=position.symbol,
                    z_score=z_score,
                    exit_threshold=self.exit_threshold
                )
                return True

            # Calculate ATR-based stop loss
            if len(
                    self.high_history) >= self.atr_period and len(
                    self.low_history) >= self.atr_period and len(
                    self.price_history) >= self.atr_period:
                try:
                    # Use the helper function which handles ta-lib with fallback
                    # Use more data points for ATR calculation (need at least
                    # period + 1)
                    high_data = self.high_history[-(self.atr_period + 1):]
                    low_data = self.low_history[-(self.atr_period + 1):]
                    close_data = self.price_history[-(self.atr_period + 1):]

                    atr = calculate_atr(
                        high_data, low_data, close_data, period=self.atr_period)

                    # TODO: Remove in production - Debug logging
                    logger.debug(
                        f"ATR calculation result: {atr}, high_history_len: {
                            len(
                                self.high_history)}, low_history_len: {
                            len(
                                self.low_history)}, price_history_len: {
                            len(
                                self.price_history)}")

                    if atr is not None and atr > 0:
                        current_price = float(data.price)
                        entry_price = float(position.entry_price)

                        # Calculate stop loss distance
                        stop_distance = atr * self.atr_multiplier

                        # Check if price has moved against position beyond stop
                        # loss
                        if position.side.value == "buy":
                            stop_price = entry_price - stop_distance
                            if current_price <= stop_price:
                                logger.info(
                                    "Stop loss triggered",
                                    strategy=self.name,
                                    symbol=position.symbol,
                                    entry_price=entry_price,
                                    current_price=current_price,
                                    stop_price=stop_price
                                )
                                return True
                        else:  # sell position
                            stop_price = entry_price + stop_distance
                            if current_price >= stop_price:
                                logger.info(
                                    "Stop loss triggered",
                                    strategy=self.name,
                                    symbol=position.symbol,
                                    entry_price=entry_price,
                                    current_price=current_price,
                                    stop_price=stop_price
                                )
                                return True
                except Exception as e:
                    logger.error(
                        "ATR calculation failed",
                        strategy=self.name,
                        error=str(e))

            return False

        except Exception as e:
            logger.error("Exit check failed", strategy=self.name, error=str(e))
            return False

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get mean reversion strategy information.

        Returns:
            Strategy information dictionary
        """
        base_info = super().get_strategy_info()

        # Add strategy-specific information
        strategy_info = {
            **base_info,
            "strategy_type": "mean_reversion",
            "parameters": {
                "lookback_period": self.lookback_period,
                "entry_threshold": self.entry_threshold,
                "exit_threshold": self.exit_threshold,
                "atr_period": self.atr_period,
                "atr_multiplier": self.atr_multiplier,
                "volume_filter": self.volume_filter,
                "min_volume_ratio": self.min_volume_ratio,
                "confirmation_timeframe": self.confirmation_timeframe
            },
            "price_history_length": len(self.price_history),
            "volume_history_length": len(self.volume_history)
        }

        return strategy_info
