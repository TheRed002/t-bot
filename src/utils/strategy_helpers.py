"""
Strategy Helper Utilities - Common functions extracted from duplicate strategy code.

This module provides shared functionality for all trading strategies to reduce code duplication
and ensure consistent behavior across the strategies module.

CRITICAL: All utilities follow the coding standards and use proper error handling.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.core.logging import get_logger
from src.core.types import MarketData
from src.utils.decimal_utils import ZERO, safe_divide, to_decimal
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class PriceHistoryManager:
    """
    Manages price history data for strategy calculations.

    This class provides a consistent interface for managing price, volume, high, and low
    historical data across all strategies.
    """

    def __init__(self, max_length: int = 200):
        """
        Initialize price history manager.

        Args:
            max_length: Maximum number of historical data points to keep
        """
        self.max_length = max_length

        # Price histories - using Decimal for financial precision
        self.price_history: list[Decimal] = []
        self.volume_history: list[Decimal] = []
        self.high_history: list[Decimal] = []
        self.low_history: list[Decimal] = []

        self._logger = get_logger(f"{__name__}.PriceHistoryManager")

    def update_history(self, data: MarketData) -> None:
        """
        Update all price histories with new market data.

        Args:
            data: Market data to add to histories
        """
        try:
            # Keep Decimal precision for financial accuracy
            price = to_decimal(data.price)
            volume = to_decimal(data.volume) if data.volume else ZERO
            high = to_decimal(data.high_price) if data.high_price else price
            low = to_decimal(data.low_price) if data.low_price else price

            # Update histories
            self.price_history.append(price)
            self.volume_history.append(volume)
            self.high_history.append(high)
            self.low_history.append(low)

            # Maintain max length
            if len(self.price_history) > self.max_length:
                self.price_history = self.price_history[-self.max_length :]
                self.volume_history = self.volume_history[-self.max_length :]
                self.high_history = self.high_history[-self.max_length :]
                self.low_history = self.low_history[-self.max_length :]

        except Exception as e:
            self._logger.error("Failed to update price history", error=str(e))

    def get_recent_prices(self, periods: int) -> list[Decimal]:
        """
        Get recent price data.

        Args:
            periods: Number of recent periods to retrieve

        Returns:
            List of recent prices
        """
        if periods <= 0:
            return []

        return (
            self.price_history[-periods:]
            if len(self.price_history) >= periods
            else self.price_history.copy()
        )

    def get_recent_volumes(self, periods: int) -> list[Decimal]:
        """
        Get recent volume data.

        Args:
            periods: Number of recent periods to retrieve

        Returns:
            List of recent volumes
        """
        if periods <= 0:
            return []

        return (
            self.volume_history[-periods:]
            if len(self.volume_history) >= periods
            else self.volume_history.copy()
        )

    def has_sufficient_data(self, required_periods: int) -> bool:
        """
        Check if we have sufficient historical data.

        Args:
            required_periods: Required number of periods

        Returns:
            True if sufficient data available
        """
        return len(self.price_history) >= required_periods

    def clear_history(self) -> None:
        """Clear all historical data."""
        self.price_history.clear()
        self.volume_history.clear()
        self.high_history.clear()
        self.low_history.clear()


class TechnicalIndicators:
    """
    Common technical indicator calculations used across strategies.

    All calculations are implemented with fallback methods to avoid external dependencies.
    """

    @staticmethod
    @time_execution
    def calculate_sma(prices: list[Decimal], period: int) -> Decimal | None:
        """
        Calculate Simple Moving Average.

        Args:
            prices: List of prices
            period: Moving average period

        Returns:
            SMA value or None if insufficient data
        """
        try:
            if len(prices) < period:
                return None

            recent_prices = prices[-period:]
            return sum(recent_prices) / to_decimal(len(recent_prices))

        except Exception as e:
            logger.error("SMA calculation failed", error=str(e))
            return None

    @staticmethod
    @time_execution
    def calculate_rsi(prices: list[Decimal], period: int = 14) -> Decimal | None:
        """
        Calculate Relative Strength Index using fallback implementation.

        Args:
            prices: List of prices
            period: RSI period (default 14)

        Returns:
            RSI value or None if insufficient data
        """
        try:
            if len(prices) < period + 1:
                return None

            # Calculate price changes
            price_changes = []
            for i in range(1, len(prices)):
                price_changes.append(prices[i] - prices[i - 1])

            if len(price_changes) < period:
                return None

            # Get recent changes
            recent_changes = price_changes[-period:]

            # Calculate gains and losses
            gains = [change for change in recent_changes if change > 0]
            losses = [-change for change in recent_changes if change < 0]

            # Calculate average gain and loss with Decimal precision
            avg_gain = sum(gains) / to_decimal(period) if gains else ZERO
            avg_loss = sum(losses) / to_decimal(period) if losses else ZERO

            # Avoid division by zero
            if avg_loss == ZERO:
                return to_decimal("100.0") if avg_gain > ZERO else to_decimal("50.0")

            # Calculate RSI
            rs = safe_divide(avg_gain, avg_loss, ZERO)
            hundred = to_decimal("100")
            one = to_decimal("1")
            rsi = hundred - safe_divide(hundred, (one + rs), hundred)

            return rsi

        except Exception as e:
            logger.error("RSI calculation failed", error=str(e))
            return None

    @staticmethod
    @time_execution
    def calculate_zscore(prices: list[Decimal], period: int) -> Decimal | None:
        """
        Calculate Z-score for mean reversion strategies.

        Args:
            prices: List of prices
            period: Lookback period for mean and standard deviation

        Returns:
            Z-score value or None if insufficient data
        """
        try:
            if len(prices) < period:
                return None

            recent_prices = prices[-period:]
            current_price = prices[-1]

            # Calculate mean and standard deviation with Decimal precision
            n_prices = to_decimal(len(recent_prices))
            mean_price = sum(recent_prices) / n_prices
            variance = sum((p - mean_price) ** 2 for p in recent_prices) / n_prices
            std_dev = variance.sqrt()

            if std_dev == ZERO:
                return ZERO

            # Calculate Z-score
            z_score = safe_divide(current_price - mean_price, std_dev, ZERO)
            return z_score

        except Exception as e:
            logger.error("Z-score calculation failed", error=str(e))
            return None

    @staticmethod
    @time_execution
    def calculate_atr(
        highs: list[Decimal], lows: list[Decimal], closes: list[Decimal], period: int = 14
    ) -> Decimal | None:
        """
        Calculate Average True Range using fallback implementation.

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: ATR period (default 14)

        Returns:
            ATR value or None if insufficient data
        """
        try:
            if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
                return None

            # Calculate true ranges
            true_ranges = []
            for i in range(1, len(highs)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i - 1])
                tr3 = abs(lows[i] - closes[i - 1])
                true_ranges.append(max(tr1, tr2, tr3))

            if len(true_ranges) < period:
                return None

            # Calculate average true range with Decimal precision
            recent_trs = true_ranges[-period:]
            atr = sum(recent_trs) / to_decimal(len(recent_trs))

            return atr

        except Exception as e:
            logger.error("ATR calculation failed", error=str(e))
            return None

    @staticmethod
    @time_execution
    def calculate_volatility(prices: list[Decimal], periods: int = 20) -> Decimal:
        """
        Calculate rolling volatility (annualized).

        Args:
            prices: List of prices
            periods: Number of periods for calculation

        Returns:
            Volatility as a percentage
        """
        try:
            if len(prices) < periods:
                return to_decimal("0.02")  # Default 2% volatility

            # Calculate returns with Decimal precision
            recent_prices = prices[-periods:]
            returns = []
            for i in range(1, len(recent_prices)):
                if recent_prices[i - 1] > ZERO:
                    ret = safe_divide(
                        recent_prices[i] - recent_prices[i - 1], recent_prices[i - 1], ZERO
                    )
                    returns.append(ret)

            if not returns:
                return to_decimal("0.02")

            # Calculate volatility (annualized) with Decimal precision
            n = to_decimal(len(returns))
            mean_return = sum(returns) / n
            variance = sum((r - mean_return) ** 2 for r in returns) / (n - to_decimal("1"))
            std_dev = variance.sqrt()
            # Annualize with sqrt(252) trading days
            volatility = std_dev * to_decimal("252").sqrt()

            return volatility

        except Exception as e:
            logger.error("Volatility calculation failed", error=str(e))
            return to_decimal("0.02")


class VolumeAnalysis:
    """
    Common volume analysis functions used across strategies.
    """

    @staticmethod
    @time_execution
    def check_volume_confirmation(
        current_volume: Decimal,
        volume_history: list[Decimal],
        lookback_period: int,
        min_ratio: Decimal = to_decimal("1.5"),
    ) -> bool:
        """
        Check if current volume meets confirmation requirements.

        Args:
            current_volume: Current period volume
            volume_history: Historical volume data
            lookback_period: Period for average calculation
            min_ratio: Minimum volume ratio required

        Returns:
            True if volume confirmation passes
        """
        try:
            if len(volume_history) < lookback_period:
                return True  # Pass if insufficient data

            if current_volume <= ZERO:
                return False

            # Calculate average volume with Decimal precision
            recent_volumes = volume_history[-lookback_period:]
            avg_volume = sum(recent_volumes) / to_decimal(len(recent_volumes))

            if avg_volume <= ZERO:
                return True  # Pass if no historical data

            # Check volume ratio
            volume_ratio = safe_divide(current_volume, avg_volume, ZERO)
            return volume_ratio >= min_ratio

        except Exception as e:
            logger.error("Volume confirmation check failed", error=str(e))
            return True  # Pass on error

    @staticmethod
    def calculate_volume_profile(
        volume_history: list[Decimal], periods: int = 20
    ) -> dict[str, Decimal]:
        """
        Calculate volume profile metrics.

        Args:
            volume_history: Historical volume data
            periods: Number of periods for analysis

        Returns:
            Dictionary with volume metrics
        """
        try:
            if len(volume_history) < periods:
                return {"avg_volume": ZERO, "volume_trend": ZERO, "volume_volatility": ZERO}

            recent_volumes = volume_history[-periods:]

            # Average volume with Decimal precision
            avg_volume = sum(recent_volumes) / to_decimal(len(recent_volumes))

            # Volume trend (slope) with Decimal precision
            volume_trend = ZERO
            if len(recent_volumes) >= 2:
                n = to_decimal(len(recent_volumes))
                x_values = [to_decimal(i) for i in range(len(recent_volumes))]
                sum_x = sum(x_values)
                sum_y = sum(recent_volumes)
                sum_xy = sum(x_values[i] * recent_volumes[i] for i in range(len(recent_volumes)))
                sum_x2 = sum(x**2 for x in x_values)

                denominator = n * sum_x2 - sum_x**2
                if denominator != ZERO:
                    volume_trend = safe_divide(n * sum_xy - sum_x * sum_y, denominator, ZERO)

            # Volume volatility with Decimal precision
            volume_volatility = ZERO
            if avg_volume > ZERO and len(recent_volumes) > 1:
                n_volumes = to_decimal(len(recent_volumes))
                volume_variance = sum((v - avg_volume) ** 2 for v in recent_volumes) / n_volumes
                volume_volatility = safe_divide(volume_variance.sqrt(), avg_volume, ZERO)

            return {
                "avg_volume": avg_volume,
                "volume_trend": volume_trend,
                "volume_volatility": volume_volatility,
            }

        except Exception as e:
            logger.error("Volume profile calculation failed", error=str(e))
            return {"avg_volume": ZERO, "volume_trend": ZERO, "volume_volatility": ZERO}


class StrategySignalValidator:
    """
    Common signal validation patterns used across strategies.
    """

    @staticmethod
    def validate_signal_metadata(
        signal_metadata: dict[str, Any], required_fields: list[str]
    ) -> bool:
        """
        Validate signal metadata contains required fields.

        Args:
            signal_metadata: Signal metadata dictionary
            required_fields: List of required field names

        Returns:
            True if all required fields present
        """
        try:
            for field in required_fields:
                if field not in signal_metadata:
                    logger.warning("Missing required field in signal metadata", field=field)
                    return False
            return True

        except Exception as e:
            logger.error("Signal metadata validation failed", error=str(e))
            return False

    @staticmethod
    def validate_price_range(
        price: Decimal, min_price: Decimal = ZERO, max_price: Decimal = to_decimal("1000000")
    ) -> bool:
        """
        Validate price is within acceptable range.

        Args:
            price: Price to validate
            min_price: Minimum acceptable price
            max_price: Maximum acceptable price

        Returns:
            True if price is valid
        """
        try:
            price_decimal = to_decimal(price) if not isinstance(price, Decimal) else price
            return min_price < price_decimal < max_price

        except Exception as e:
            logger.error("Price range validation failed", error=str(e))
            return False

    @staticmethod
    def check_signal_freshness(signal_timestamp: datetime, max_age_seconds: int = 300) -> bool:
        """
        Check if signal is fresh enough for execution.

        Args:
            signal_timestamp: Signal creation timestamp
            max_age_seconds: Maximum age in seconds

        Returns:
            True if signal is fresh enough
        """
        try:
            now = datetime.now(signal_timestamp.tzinfo)
            age = (now - signal_timestamp).total_seconds()
            return age <= max_age_seconds

        except Exception as e:
            logger.error("Signal freshness check failed", error=str(e))
            return False
