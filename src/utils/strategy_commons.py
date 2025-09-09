"""
Strategy Commons - Shared utility class for common strategy operations.

This module provides a comprehensive utility class that strategies can use to access
common functionality without duplicating code.

CRITICAL: All utilities follow the coding standards and integrate with existing services.
"""

from decimal import Decimal
from typing import Any

from src.core.logging import get_logger
from src.core.types import MarketData, Position, Signal
from src.utils.arbitrage_helpers import FeeCalculator, OpportunityAnalyzer, SpreadAnalyzer
from src.utils.decorators import time_execution
from src.utils.strategy_helpers import (
    PriceHistoryManager,
    StrategySignalValidator,
    TechnicalIndicators,
    VolumeAnalysis,
)

logger = get_logger(__name__)


class StrategyCommons:
    """
    Comprehensive utility class providing common strategy operations.

    This class aggregates all common strategy functionality into a single interface
    that strategies can use to avoid code duplication.
    """

    def __init__(self, strategy_name: str, config: dict[str, Any] | None = None):
        """
        Initialize strategy commons.

        Args:
            strategy_name: Name of the strategy using this commons
            config: Configuration dictionary
        """
        self.strategy_name = strategy_name
        self.config = config or {}
        self._logger = get_logger(f"{__name__}.StrategyCommons_{strategy_name}")

        # Initialize components
        max_history = self.config.get("max_history_length", 200)
        self.price_history = PriceHistoryManager(max_history)

        # Technical indicators
        self.indicators = TechnicalIndicators()

        # Volume analysis
        self.volume_analysis = VolumeAnalysis()

        # Signal validation
        self.signal_validator = StrategySignalValidator()

        # Arbitrage helpers (if needed)
        self.fee_calculator = FeeCalculator()
        self.opportunity_analyzer = OpportunityAnalyzer()
        self.spread_analyzer = SpreadAnalyzer()

    def update_market_data(self, data: MarketData) -> None:
        """
        Update all historical data with new market data.

        Args:
            data: New market data
        """
        try:
            self.price_history.update_history(data)

        except Exception as e:
            self._logger.error("Failed to update market data", error=str(e))

    def get_technical_analysis(
        self, indicator_type: str, period: int = 14, **kwargs
    ) -> float | None:
        """
        Get technical indicator value.

        Args:
            indicator_type: Type of indicator (sma, rsi, zscore, atr, volatility)
            period: Period for calculation
            **kwargs: Additional parameters for specific indicators

        Returns:
            Indicator value or None if insufficient data
        """
        try:
            if indicator_type == "sma":
                prices = self.price_history.get_recent_prices(period + 5)
                return self.indicators.calculate_sma(prices, period)

            elif indicator_type == "rsi":
                prices = self.price_history.get_recent_prices(period + 10)
                return self.indicators.calculate_rsi(prices, period)

            elif indicator_type == "zscore":
                prices = self.price_history.get_recent_prices(period + 5)
                return self.indicators.calculate_zscore(prices, period)

            elif indicator_type == "atr":
                highs = self.price_history.high_history[-(period + 5) :]
                lows = self.price_history.low_history[-(period + 5) :]
                closes = self.price_history.get_recent_prices(period + 5)
                return self.indicators.calculate_atr(highs, lows, closes, period)

            elif indicator_type == "volatility":
                prices = self.price_history.get_recent_prices(kwargs.get("vol_periods", 20))
                return self.indicators.calculate_volatility(prices, kwargs.get("vol_periods", 20))

            else:
                self._logger.warning("Unknown indicator type", indicator_type=indicator_type)
                return None

        except Exception as e:
            self._logger.error("Technical analysis failed", indicator=indicator_type, error=str(e))
            return None

    def check_volume_confirmation(
        self, current_volume: float, lookback_period: int = 20, min_ratio: float = 1.5
    ) -> bool:
        """
        Check if current volume meets confirmation requirements.

        Args:
            current_volume: Current volume to check
            lookback_period: Period for average calculation
            min_ratio: Minimum volume ratio required

        Returns:
            True if volume confirmation passes
        """
        try:
            volume_history = self.price_history.get_recent_volumes(lookback_period + 5)
            return self.volume_analysis.check_volume_confirmation(
                current_volume, volume_history, lookback_period, min_ratio
            )

        except Exception as e:
            self._logger.error("Volume confirmation check failed", error=str(e))
            return True  # Pass on error

    def get_volume_profile(self, periods: int = 20) -> dict[str, float]:
        """
        Get volume profile metrics.

        Args:
            periods: Number of periods for analysis

        Returns:
            Volume profile metrics
        """
        try:
            volume_history = self.price_history.get_recent_volumes(periods + 5)
            return self.volume_analysis.calculate_volume_profile(volume_history, periods)

        except Exception as e:
            self._logger.error("Volume profile calculation failed", error=str(e))
            return {"avg_volume": 0.0, "volume_trend": 0.0, "volume_volatility": 0.0}

    def validate_signal_comprehensive(
        self, signal: Signal, required_metadata: list[str] | None = None, max_age_seconds: int = 300
    ) -> bool:
        """
        Perform comprehensive signal validation.

        Args:
            signal: Signal to validate
            required_metadata: List of required metadata fields
            max_age_seconds: Maximum signal age in seconds

        Returns:
            True if signal passes all validations
        """
        try:
            # Check signal freshness
            if not self.signal_validator.check_signal_freshness(signal.timestamp, max_age_seconds):
                return False

            # Validate metadata if specified
            if required_metadata:
                if not hasattr(signal, "metadata") or not signal.metadata:
                    return False
                if not self.signal_validator.validate_signal_metadata(
                    signal.metadata, required_metadata
                ):
                    return False

            # Validate signal strength
            if signal.strength < 0.0 or signal.strength > 1.0:
                self._logger.warning("Invalid signal strength", strength=signal.strength)
                return False

            return True

        except Exception as e:
            self._logger.error("Signal validation failed", error=str(e))
            return False

    def calculate_position_size_with_risk(
        self,
        signal: Signal,
        total_capital: Decimal,
        base_risk_pct: float = 0.02,
        max_position_pct: float = 0.1,
    ) -> Decimal:
        """
        Calculate position size with risk management.

        Args:
            signal: Trading signal
            total_capital: Total available capital
            base_risk_pct: Base risk percentage (0-1)
            max_position_pct: Maximum position percentage (0-1)

        Returns:
            Calculated position size
        """
        try:
            # Base position size
            base_size = total_capital * Decimal(str(base_risk_pct))

            # Adjust by signal strength
            strength_adjusted = base_size * Decimal(str(signal.strength))

            # Apply maximum limit
            max_size = total_capital * Decimal(str(max_position_pct))
            final_size = min(strength_adjusted, max_size)

            # Ensure minimum size
            min_size = total_capital * Decimal("0.001")  # 0.1%
            final_size = max(final_size, min_size)

            return final_size

        except Exception as e:
            self._logger.error("Position size calculation failed", error=str(e))
            return total_capital * Decimal(str(base_risk_pct * 0.5))  # Conservative fallback

    def check_stop_loss_conditions(
        self,
        position: Position,
        current_price: Decimal,
        stop_loss_pct: float = 0.02,
        use_atr: bool = True,
        atr_multiplier: float = 2.0,
    ) -> bool:
        """
        Check if stop loss conditions are met.

        Args:
            position: Current position
            current_price: Current market price
            stop_loss_pct: Stop loss percentage (0-1)
            use_atr: Whether to use ATR for dynamic stop loss
            atr_multiplier: ATR multiplier for stop distance

        Returns:
            True if stop loss should be triggered
        """
        try:
            entry_price = position.entry_price

            # Calculate stop distance
            if use_atr and self.price_history.has_sufficient_data(20):
                atr = self.get_technical_analysis("atr", period=14)
                if atr and atr > 0:
                    stop_distance = Decimal(str(atr * atr_multiplier))
                else:
                    # Fallback to percentage
                    stop_distance = entry_price * Decimal(str(stop_loss_pct))
            else:
                stop_distance = entry_price * Decimal(str(stop_loss_pct))

            # Check stop loss based on position direction
            if hasattr(position, "side") and position.side:
                if position.side.value.lower() == "buy" or position.quantity > 0:
                    # Long position - check downside
                    stop_price = entry_price - stop_distance
                    return current_price <= stop_price
                else:
                    # Short position - check upside
                    stop_price = entry_price + stop_distance
                    return current_price >= stop_price
            else:
                # Determine direction from quantity
                if position.quantity > 0:
                    # Long position
                    stop_price = entry_price - stop_distance
                    return current_price <= stop_price
                else:
                    # Short position
                    stop_price = entry_price + stop_distance
                    return current_price >= stop_price

        except Exception as e:
            self._logger.error("Stop loss check failed", error=str(e))
            return False

    def calculate_take_profit_level(
        self,
        entry_price: Decimal,
        position_side: str,
        target_profit_pct: float = 0.04,
        use_atr: bool = True,
        atr_multiplier: float = 3.0,
    ) -> Decimal:
        """
        Calculate take profit level.

        Args:
            entry_price: Position entry price
            position_side: Position side ("buy" or "sell")
            target_profit_pct: Target profit percentage
            use_atr: Whether to use ATR for dynamic targets
            atr_multiplier: ATR multiplier for target distance

        Returns:
            Take profit price level
        """
        try:
            # Calculate target distance
            if use_atr and self.price_history.has_sufficient_data(20):
                atr = self.get_technical_analysis("atr", period=14)
                if atr and atr > 0:
                    target_distance = Decimal(str(atr * atr_multiplier))
                else:
                    target_distance = entry_price * Decimal(str(target_profit_pct))
            else:
                target_distance = entry_price * Decimal(str(target_profit_pct))

            # Calculate take profit level
            if position_side.lower() == "buy":
                take_profit = entry_price + target_distance
            else:
                take_profit = entry_price - target_distance

            return take_profit

        except Exception as e:
            self._logger.error("Take profit calculation failed", error=str(e))
            # Fallback calculation
            if position_side.lower() == "buy":
                return entry_price * (Decimal("1") + Decimal(str(target_profit_pct)))
            else:
                return entry_price * (Decimal("1") - Decimal(str(target_profit_pct)))

    def get_market_condition_summary(self) -> dict[str, Any]:
        """
        Get comprehensive market condition summary.

        Returns:
            Dictionary with market condition metrics
        """
        try:
            if not self.price_history.has_sufficient_data(20):
                return {"condition": "insufficient_data", "confidence": 0.0}

            # Get technical indicators
            rsi = self.get_technical_analysis("rsi", period=14)
            volatility = self.get_technical_analysis("volatility")
            sma_20 = self.get_technical_analysis("sma", period=20)
            sma_50 = self.get_technical_analysis("sma", period=50)

            # Current price
            current_price = (
                self.price_history.price_history[-1] if self.price_history.price_history else 0
            )

            # Volume profile
            volume_profile = self.get_volume_profile()

            # Determine market condition
            condition = "neutral"
            confidence = 0.5

            if rsi and sma_20 and sma_50:
                if rsi > 70 and current_price > sma_20 > sma_50:
                    condition = "overbought_uptrend"
                    confidence = 0.8
                elif rsi < 30 and current_price < sma_20 < sma_50:
                    condition = "oversold_downtrend"
                    confidence = 0.8
                elif current_price > sma_20 > sma_50:
                    condition = "uptrend"
                    confidence = 0.7
                elif current_price < sma_20 < sma_50:
                    condition = "downtrend"
                    confidence = 0.7

            return {
                "condition": condition,
                "confidence": confidence,
                "rsi": rsi,
                "volatility": volatility,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "current_price": current_price,
                "volume_profile": volume_profile,
            }

        except Exception as e:
            self._logger.error("Market condition analysis failed", error=str(e))
            return {"condition": "error", "confidence": 0.0}

    @time_execution
    def analyze_trend_strength(self, lookback_period: int = 20) -> dict[str, float]:
        """
        Analyze trend strength using multiple indicators.

        Args:
            lookback_period: Period for trend analysis

        Returns:
            Dictionary with trend metrics
        """
        try:
            if not self.price_history.has_sufficient_data(lookback_period * 2):
                return {"trend_strength": 0.0, "direction": 0.0, "consistency": 0.0}

            # Get moving averages
            sma_short = self.get_technical_analysis("sma", period=lookback_period // 2)
            sma_long = self.get_technical_analysis("sma", period=lookback_period)

            # Current price
            current_price = self.price_history.price_history[-1]

            # Calculate trend metrics
            trend_strength = 0.0
            direction = 0.0
            consistency = 0.0

            if sma_short and sma_long:
                # Direction (1 = up, -1 = down, 0 = sideways)
                if current_price > sma_short > sma_long:
                    direction = 1.0
                elif current_price < sma_short < sma_long:
                    direction = -1.0

                # Strength based on separation
                if sma_long > 0:
                    separation = abs(sma_short - sma_long) / sma_long
                    trend_strength = min(separation * 10, 1.0)  # Scale to 0-1

                # Consistency based on price vs MA alignment
                recent_prices = self.price_history.get_recent_prices(lookback_period // 2)
                if recent_prices and sma_short:
                    aligned_count = sum(
                        1
                        for price in recent_prices
                        if (direction > 0 and price >= sma_short)
                        or (direction < 0 and price <= sma_short)
                    )
                    consistency = aligned_count / len(recent_prices)

            return {
                "trend_strength": trend_strength,
                "direction": direction,
                "consistency": consistency,
            }

        except Exception as e:
            self._logger.error("Trend strength analysis failed", error=str(e))
            return {"trend_strength": 0.0, "direction": 0.0, "consistency": 0.0}

    def cleanup(self) -> None:
        """Clean up resources and clear histories."""
        try:
            self.price_history.clear_history()
            self._logger.info("Strategy commons cleanup completed")

        except Exception as e:
            self._logger.error("Cleanup failed", error=str(e))
