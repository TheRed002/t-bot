"""
Mean Reversion Strategy Implementation.

This module implements a statistical arbitrage strategy based on mean reversion principles.
The strategy identifies overbought/oversold conditions using Z-score calculations and generates
signals when prices deviate significantly from their moving average.

CRITICAL: This strategy MUST inherit from BaseStrategy and follow the exact interface.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from src.core.caching.cache_decorators import cached
from src.core.caching.cache_manager import get_cache_manager

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

# Monitoring and caching integration
# From P-008+ - Use risk management
# MANDATORY: Import from P-011 - NEVER recreate the base strategy
from src.strategies.base import BaseStrategy
from src.strategies.dependencies import StrategyServiceContainer
from src.utils.datetime_utils import to_timestamp
from src.utils.decimal_utils import safe_divide

# From P-007A - Use decorators and validators
from src.utils.decorators import time_execution
from src.utils.formatters import format_currency, format_percentage
from src.utils.strategy_commons import StrategyCommons
from src.utils.validation.market_data_validation import MarketDataValidator
from src.utils.validators import ValidationFramework


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

    def __init__(self, config: dict[str, Any], services: "StrategyServiceContainer | None" = None):
        """Initialize Mean Reversion Strategy.

        Args:
            config: Strategy configuration dictionary
            services: Service container with all required dependencies
        """
        super().__init__(config, services)
        # Use the name from config (already set by BaseStrategy)

        # Strategy-specific parameters with defaults
        self.lookback_period = self.config.parameters.get("lookback_period", 20)
        self.entry_threshold = self.config.parameters.get("entry_threshold", 2.0)
        self.exit_threshold = self.config.parameters.get("exit_threshold", 0.5)
        self.atr_period = self.config.parameters.get("atr_period", 14)
        self.atr_multiplier = self.config.parameters.get("atr_multiplier", 2.0)
        self.volume_filter = self.config.parameters.get("volume_filter", True)
        self.min_volume_ratio = self.config.parameters.get("min_volume_ratio", 1.5)
        self.confirmation_timeframe = self.config.parameters.get("confirmation_timeframe", "1h")

        # Initialize strategy commons for shared functionality
        self.commons = StrategyCommons(
            self.name, {"max_history_length": max(self.lookback_period, self.atr_period) + 20}
        )

        # Initialize cache manager for performance
        self.cache_manager = get_cache_manager()

        # Initialize market data validator
        self.market_data_validator = MarketDataValidator()

        # Initialize metrics collector for telemetry
        self._metrics_collector = None
        if services and services.monitoring_service:
            self._metrics_collector = services.monitoring_service.get_metrics_collector()

        # Initialize analytics service for strategy performance tracking
        self._analytics_service = None
        if services and services.analytics_service:
            self._analytics_service = services.analytics_service
            self.logger.info("Analytics service integrated for strategy performance tracking", strategy=self.name)

        # Store symbol for indicator calculations
        self._current_symbol: "str | None" = None

        self.logger.info(
            "Mean Reversion Strategy initialized",
            strategy=self.name,
            lookback_period=self.lookback_period,
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
        )

    @property
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        return StrategyType.MEAN_REVERSION


    @cached(ttl=30, namespace="strategy", key_generator=lambda data: f"signals_{data.symbol}_{data.timestamp}")
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @with_error_context(operation="mean_reversion_signal_generation")
    @with_retry(max_attempts=3, base_delay=Decimal("1.0"))
    @time_execution
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """Generate mean reversion signals from market data.

        MANDATORY: Use graceful error handling and input validation.

        Args:
            data: Market data for signal generation

        Returns:
            List of trading signals
        """
        try:
            # MANDATORY: Enhanced input validation using market data validator
            validation_result = await self.market_data_validator.validate_market_data(data)
            if not validation_result.is_valid:
                self.logger.warning(
                    "Market data validation failed",
                    strategy=self.name,
                    errors=validation_result.errors
                )
                return []

            # Additional symbol validation
            try:
                validated_symbol = ValidationFramework.validate_symbol(data.symbol)
            except (ValidationError, ValueError) as e:
                self.logger.warning("Invalid symbol format", strategy=self.name, symbol=data.symbol, error=str(e))
                return []

            # Validate price range
            price = float(data.price)
            try:
                validated_price = ValidationFramework.validate_price(price)
            except (ValidationError, ValueError) as e:
                self.logger.warning("Price out of acceptable range", strategy=self.name, price=price, error=str(e))
                return []

            # Store current symbol for indicator calculations
            self._current_symbol = data.symbol

            # Calculate indicators using shared BaseStrategy methods
            sma = await self.get_sma(data.symbol, self.lookback_period)
            if sma is None:
                self.logger.debug(
                    "Insufficient data for SMA calculation",
                    strategy=self.name,
                    symbol=data.symbol,
                    period=self.lookback_period,
                )
                return []

            # Calculate standard deviation for Z-score
            volatility = await self.get_volatility(data.symbol, self.lookback_period)
            if volatility is None or volatility == 0:
                self.logger.debug(
                    "Cannot calculate volatility for Z-score",
                    strategy=self.name,
                    symbol=data.symbol,
                )
                return []

            # Calculate Z-score using safe decimal operations
            current_price = Decimal(str(price))
            price_deviation = current_price - sma
            z_score = safe_divide(price_deviation, volatility, default=Decimal("0"))

            # Record metrics if available
            if self._metrics_collector:
                self._metrics_collector.record_gauge(
                    "strategy_z_score",
                    z_score,
                    labels={"strategy": self.name, "symbol": data.symbol}
                )

            # Check volume filter if enabled
            if self.volume_filter:
                volume_ratio = await self.get_volume_ratio(data.symbol, self.lookback_period)
                if volume_ratio is None or volume_ratio < Decimal(str(self.min_volume_ratio)):
                    self.logger.debug(
                        "Volume filter rejected signal",
                        strategy=self.name,
                        z_score=z_score,
                        volume_ratio=format_percentage(float(volume_ratio or 0))
                    )
                    # Record filtered signal metric
                    if self._metrics_collector:
                        self._metrics_collector.increment_counter(
                            "strategy_signals_filtered",
                            labels={"strategy": self.name, "reason": "volume"}
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
                    strength=Decimal(str(confidence)),
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    source=self.name,
                    metadata={
                        "z_score": z_score,
                        "entry_threshold": self.entry_threshold,
                        "lookback_period": self.lookback_period,
                        "signal_type": "entry",
                        "sma_value": float(sma),
                        "volatility": float(volatility),
                        "current_price": format_currency(float(current_price)),
                        "timestamp_formatted": to_timestamp(data.timestamp),
                    },
                )

                if await self.validate_signal(signal):
                    signals.append(signal)
                    # Record signal generation metric
                    if self._metrics_collector:
                        self._metrics_collector.increment_counter(
                            "strategy_signals_generated",
                            labels={
                                "strategy": self.name,
                                "symbol": data.symbol,
                                "signal_type": "entry",
                                "direction": direction.value
                            }
                        )

                    # Record strategy analytics
                    if self._analytics_service:
                        await self._record_strategy_analytics(signal, data, {
                            "z_score": z_score,
                            "volatility": float(volatility),
                            "sma": float(sma),
                            "signal_quality": "high" if confidence > 0.8 else "medium"
                        })

                    self.logger.info(
                        "Mean reversion entry signal generated",
                        strategy=self.name,
                        symbol=data.symbol,
                        direction=direction.value,
                        z_score=z_score,
                        confidence=format_percentage(confidence),
                        formatted_price=format_currency(float(current_price)),
                    )

            # Exit signals for existing positions
            if abs(z_score) <= self.exit_threshold:
                # Generate exit signals for both directions
                for direction in [SignalDirection.BUY, SignalDirection.SELL]:
                    # For exit signals, confidence should be high when Z-score is close to zero
                    # Use a different confidence calculation for exit signals
                    confidence = max(0.8, 1.0 - (abs(z_score) / self.exit_threshold))

                    signal = Signal(
                        direction=direction,
                        strength=Decimal(str(confidence)),
                        timestamp=data.timestamp,
                        symbol=data.symbol,
                        source=self.name,
                        metadata={
                            "z_score": z_score,
                            "exit_threshold": self.exit_threshold,
                            "signal_type": "exit",
                        },
                    )

                    if await self.validate_signal(signal):
                        signals.append(signal)
                        self.logger.info(
                            "Mean reversion exit signal generated",
                            strategy=self.name,
                            symbol=data.symbol,
                            direction=direction.value,
                            z_score=z_score,
                            confidence=confidence,
                        )

            return signals

        except Exception as e:
            # Record error metric
            if self._metrics_collector:
                self._metrics_collector.increment_counter(
                    "strategy_errors",
                    labels={
                        "strategy": self.name,
                        "operation": "signal_generation",
                        "error_type": type(e).__name__
                    }
                )

            self.logger.error(
                "Signal generation failed",
                strategy=self.name,
                error=str(e),
                symbol=data.symbol if data else "unknown",
                exc_info=True,
            )
            return []  # MANDATORY: Graceful degradation

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate mean reversion signal before execution.

        MANDATORY: Check signal confidence, direction, timestamp

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Basic signal validation - use min_confidence from parameters
            min_confidence = self.config.parameters.get("min_confidence", 0.6)
            if not signal or signal.strength < Decimal(str(min_confidence)):
                self.logger.debug(
                    "Signal confidence below threshold",
                    strategy=self.name,
                    confidence=signal.strength if signal else 0,
                    min_confidence=min_confidence,
                )
                return False

            # Check if signal is too old (more than 5 minutes)
            if datetime.now(signal.timestamp.tzinfo) - signal.timestamp > timedelta(minutes=5):
                self.logger.debug("Signal too old", strategy=self.name, signal_age_minutes=5)
                return False

            # Validate signal metadata
            metadata = signal.metadata
            if "z_score" not in metadata:
                self.logger.warning("Missing z_score in signal metadata", strategy=self.name)
                return False

            z_score = metadata.get("z_score")
            if not isinstance(z_score, (int, float)):
                self.logger.warning(
                    "Invalid z_score type", strategy=self.name, z_score_type=type(z_score)
                )
                return False

            # Additional validation for entry signals
            if metadata.get("signal_type") == "entry":
                if abs(z_score) < self.entry_threshold:
                    self.logger.debug(
                        "Entry signal z_score below threshold",
                        strategy=self.name,
                        z_score=z_score,
                        threshold=self.entry_threshold,
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error("Signal validation failed", strategy=self.name, error=str(e))
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
            base_size = Decimal(str(self.config.position_size_pct or 0.02))

            # Adjust based on signal confidence
            confidence_factor = signal.strength

            # Adjust based on Z-score magnitude (stronger deviation = larger
            # position)
            metadata = signal.metadata
            z_score = abs(metadata.get("z_score", 0))
            z_score_factor = min(z_score / self.entry_threshold, 2.0)  # Cap at 2x

            # Calculate final position size
            position_size = (
                base_size * Decimal(str(confidence_factor)) * Decimal(str(z_score_factor))
            )

            # Ensure position size is within limits
            max_size = Decimal(str(self.config.parameters.get("max_position_size_pct", 0.1)))
            position_size = min(position_size, max_size)

            self.logger.debug(
                "Position size calculated",
                strategy=self.name,
                base_size=float(base_size),
                confidence_factor=confidence_factor,
                z_score_factor=z_score_factor,
                final_size=float(position_size),
            )

            return position_size

        except Exception as e:
            self.logger.error("Position size calculation failed", strategy=self.name, error=str(e))
            # Return minimum position size on error
            return Decimal(str((self.config.position_size_pct or 0.02) * 0.5))

    async def should_exit(self, position: Position, data: MarketData) -> bool:
        """Determine if position should be closed.

        Args:
            position: Current position
            data: Current market data

        Returns:
            True if position should be closed, False otherwise
        """
        try:
            # Calculate current indicators
            sma = await self.get_sma(position.symbol, self.lookback_period)
            volatility = await self.get_volatility(position.symbol, self.lookback_period)

            if sma is None or volatility is None or volatility == 0:
                return False

            # Calculate current Z-score
            current_price = data.price if isinstance(data.price, Decimal) else Decimal(str(data.price))
            z_score = (current_price - sma) / volatility

            # Exit if Z-score is within exit threshold
            if abs(z_score) <= self.exit_threshold:
                self.logger.info(
                    "Exit signal triggered by Z-score",
                    strategy=self.name,
                    symbol=position.symbol,
                    z_score=z_score,
                    exit_threshold=self.exit_threshold,
                )
                return True

            # Check ATR-based stop loss using service
            atr = await self.get_atr(position.symbol, self.atr_period)
            if atr is not None and atr > 0:
                stop_distance = atr * Decimal(str(self.atr_multiplier))
                entry_price = position.entry_price
                current_price_decimal = data.price if isinstance(data.price, Decimal) else Decimal(str(data.price))

                if position.side.value == "LONG":
                    stop_price = entry_price - stop_distance
                    if current_price_decimal <= stop_price:
                        self.logger.info(
                            "ATR stop loss triggered",
                            strategy=self.name,
                            symbol=position.symbol,
                            current_price=float(current_price_decimal),
                        )
                        return True
                else:  # SHORT
                    stop_price = entry_price + stop_distance
                    if current_price_decimal >= stop_price:
                        self.logger.info(
                            "ATR stop loss triggered",
                            strategy=self.name,
                            symbol=position.symbol,
                            current_price=float(current_price_decimal),
                        )
                        return True

            return False

        except Exception as e:
            self.logger.error("Exit check failed", strategy=self.name, error=str(e))
            return False

    def get_strategy_info(self) -> dict[str, Any]:
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
                "confirmation_timeframe": self.confirmation_timeframe,
            },
            "current_symbol": self._current_symbol,
        }

        return strategy_info

    # NOTE: Technical indicator methods are now inherited from BaseStrategy
    # This eliminates code duplication across all strategies

    async def _record_strategy_analytics(
        self, signal: Signal, market_data: MarketData, metrics: dict[str, Any]
    ) -> None:
        """Record strategy analytics for performance tracking and analysis.

        Args:
            signal: Generated trading signal
            market_data: Market data that triggered the signal
            metrics: Additional strategy-specific metrics
        """
        try:
            if not self._analytics_service:
                return

            # Record strategy event with operational analytics
            if hasattr(self._analytics_service, "operational_service"):
                operational_service = self._analytics_service.operational_service
                if operational_service:
                    operational_service.record_strategy_event(
                        strategy_name=self.name,
                        event_type="signal_generated",
                        success=True,
                        signal_direction=signal.direction.value,
                        signal_strength=float(signal.strength),
                        symbol=signal.symbol,
                        z_score=metrics.get("z_score"),
                        volatility=metrics.get("volatility"),
                        signal_quality=metrics.get("signal_quality"),
                        entry_threshold=self.entry_threshold,
                        lookback_period=self.lookback_period
                    )

            # Create and record strategy metrics for performance analysis
            from src.analytics.types import StrategyMetrics

            strategy_metrics = StrategyMetrics(
                strategy_name=self.name,
                strategy_type=self.strategy_type.value,
                timestamp=market_data.timestamp,
                total_signals=1,  # This signal
                profitable_signals=0,  # Will be updated by trade results
                win_rate=0.0,  # Will be calculated later
                avg_return=0.0,  # Will be calculated later
                max_drawdown=0.0,  # Will be calculated later
                sharpe_ratio=0.0,  # Will be calculated later
                alpha=0.0,  # Will be calculated later
                beta=0.0,  # Will be calculated later
                metadata={
                    "current_signal": {
                        "direction": signal.direction.value,
                        "strength": float(signal.strength),
                        "z_score": metrics.get("z_score"),
                        "volatility": metrics.get("volatility"),
                        "sma": metrics.get("sma"),
                        "signal_quality": metrics.get("signal_quality")
                    },
                    "strategy_params": {
                        "lookback_period": self.lookback_period,
                        "entry_threshold": self.entry_threshold,
                        "exit_threshold": self.exit_threshold,
                        "atr_multiplier": self.atr_multiplier
                    }
                }
            )

            # Record strategy metrics for analysis
            if hasattr(self._analytics_service, "realtime_analytics"):
                realtime_service = self._analytics_service.realtime_analytics
                if realtime_service and hasattr(realtime_service, "record_strategy_metrics"):
                    await realtime_service.record_strategy_metrics(strategy_metrics)

            self.logger.debug(
                "Strategy analytics recorded",
                strategy=self.name,
                symbol=signal.symbol,
                signal_quality=metrics.get("signal_quality"),
                z_score=metrics.get("z_score")
            )

        except Exception as e:
            self.logger.warning(
                "Failed to record strategy analytics",
                strategy=self.name,
                error=str(e),
                symbol=signal.symbol
            )
