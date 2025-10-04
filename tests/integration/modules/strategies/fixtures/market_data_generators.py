"""
Realistic Market Data Generators for Strategy Testing.

This module provides mathematically consistent market data for testing
real technical indicators and strategy signals. The data generators
create realistic price action patterns that can be used to validate
indicator calculations and strategy behavior.

Key Features:
- Realistic OHLCV data with proper price relationships
- Trend, consolidation, and reversal patterns
- Volume correlation with price movements
- Multi-timeframe data generation
- Deterministic patterns for reproducible testing
"""

import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Generator, List

from src.core.types import MarketData


class MarketRegime(Enum):
    """Market regime types for realistic data generation."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CONSOLIDATING = "consolidating"
    VOLATILE = "volatile"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"


class MarketDataGenerator:
    """
    Generates realistic market data for testing technical indicators.

    Creates mathematically consistent OHLCV data that can be used to
    validate technical indicator calculations and strategy signals.
    """

    def __init__(self, seed: int = 42):
        """Initialize generator with optional seed for reproducibility."""
        self.seed = seed
        random.seed(seed)

        # Default parameters
        self.base_price = Decimal("50000.00")
        self.base_volume = Decimal("500.00")
        self.volatility = Decimal("0.02")  # 2% daily volatility

    def generate_trending_data(
        self,
        symbol: str = "BTC/USDT",
        periods: int = 100,
        trend_strength: float = 0.001,  # 0.1% per period
        direction: int = 1,  # 1 for up, -1 for down
    ) -> List[MarketData]:
        """
        Generate trending market data with realistic price action.

        Args:
            symbol: Trading symbol
            periods: Number of periods to generate
            trend_strength: Trend strength per period (0.001 = 0.1%)
            direction: 1 for uptrend, -1 for downtrend

        Returns:
            List of MarketData with realistic trending pattern
        """
        # Reset random seed for reproducibility across multiple calls
        random.seed(self.seed)

        data = []
        current_price = self.base_price
        current_time = datetime.now(timezone.utc) - timedelta(hours=periods)

        for i in range(periods):
            # Add trend component (convert all to Decimal to avoid NaN)
            direction_dec = Decimal(str(direction))
            trend_str_dec = Decimal(str(trend_strength))
            random_mult = Decimal(str(1 + 0.5 * random.random()))
            trend_factor = direction_dec * trend_str_dec * random_mult

            # Add random noise
            noise_factor = Decimal(str(random.uniform(-0.005, 0.005)))

            # Calculate new price with bounds checking
            price_change = current_price * (trend_factor + noise_factor)
            new_price = current_price + price_change

            # Prevent invalid prices
            if new_price <= Decimal("0") or new_price.is_nan():
                new_price = current_price * Decimal("0.999")

            # Generate realistic OHLC
            volatility_range = new_price * self.volatility * Decimal(str(random.uniform(0.3, 1.0)))

            # Validate volatility_range
            if volatility_range.is_nan() or volatility_range <= Decimal("0"):
                volatility_range = new_price * Decimal("0.001")

            open_price = current_price
            close_price = new_price

            # High and low based on intraperiod volatility
            high_offset = volatility_range * Decimal(str(random.uniform(0.3, 0.8)))
            low_offset = volatility_range * Decimal(str(random.uniform(0.3, 0.8)))

            high_price = max(open_price, close_price) + high_offset
            low_price = min(open_price, close_price) - low_offset

            # Validate OHLC values
            if high_price.is_nan() or high_price <= Decimal("0"):
                high_price = close_price * Decimal("1.001")
            if low_price.is_nan() or low_price <= Decimal("0"):
                low_price = close_price * Decimal("0.999")

            # Generate correlated volume (higher on larger moves)
            # Prevent division by zero
            if open_price > Decimal("0"):
                price_move_pct = abs((close_price - open_price) / open_price)
            else:
                price_move_pct = Decimal("0.001")

            volume_multiplier = Decimal("1.0") + price_move_pct * Decimal("5.0")
            volume = self.base_volume * volume_multiplier * Decimal(str(random.uniform(0.5, 2.0)))

            # Validate volume
            if volume.is_nan() or volume <= Decimal("0"):
                volume = self.base_volume

            # Quantize all prices to 18 decimal places for crypto precision
            market_data = MarketData(
                symbol=symbol,
                open=open_price.quantize(Decimal("0.000000000000000001")),
                high=high_price.quantize(Decimal("0.000000000000000001")),
                low=low_price.quantize(Decimal("0.000000000000000001")),
                close=close_price.quantize(Decimal("0.000000000000000001")),
                volume=volume.quantize(Decimal("0.000000000000000001")),
                exchange="binance",
                timestamp=current_time + timedelta(hours=i),
                bid_price=(close_price - Decimal("0.50")).quantize(Decimal("0.000000000000000001")),
                ask_price=(close_price + Decimal("0.50")).quantize(Decimal("0.000000000000000001")),
            )

            data.append(market_data)
            current_price = close_price

        return data

    def generate_consolidating_data(
        self,
        symbol: str = "BTC/USDT",
        periods: int = 100,
        range_pct: float = 0.02,  # 2% trading range
    ) -> List[MarketData]:
        """
        Generate consolidating (sideways) market data.

        Args:
            symbol: Trading symbol
            periods: Number of periods to generate
            range_pct: Price range as percentage of base price

        Returns:
            List of MarketData with consolidation pattern
        """
        # Reset random seed for reproducibility across multiple calls
        random.seed(self.seed)

        data = []
        center_price = self.base_price
        range_amount = center_price * Decimal(str(range_pct))
        upper_bound = center_price + range_amount / 2
        lower_bound = center_price - range_amount / 2

        current_time = datetime.now(timezone.utc) - timedelta(hours=periods)

        for i in range(periods):
            # Oscillate around center with mean reversion
            current_distance_from_center = (data[-1].close - center_price) if data else Decimal("0")
            mean_reversion_force = -current_distance_from_center * Decimal("0.1")

            # Random component
            random_component = Decimal(str(random.uniform(-0.003, 0.003))) * center_price

            # Calculate price change
            price_change = mean_reversion_force + random_component

            if data:
                open_price = data[-1].close
            else:
                open_price = center_price

            close_price = open_price + price_change

            # Keep within bounds
            close_price = max(lower_bound, min(upper_bound, close_price))

            # Generate realistic OHLC within the range
            intraperiod_range = center_price * Decimal("0.005")  # 0.5% intraperiod range

            high_price = max(open_price, close_price) + intraperiod_range * Decimal(str(random.random()))
            low_price = min(open_price, close_price) - intraperiod_range * Decimal(str(random.random()))

            # Ensure bounds are respected
            high_price = min(high_price, upper_bound * Decimal("1.001"))
            low_price = max(low_price, lower_bound * Decimal("0.999"))

            # Lower volume during consolidation
            volume = self.base_volume * Decimal(str(random.uniform(0.3, 0.8)))

            # Quantize all prices to 18 decimal places for crypto precision
            market_data = MarketData(
                symbol=symbol,
                open=open_price.quantize(Decimal("0.000000000000000001")),
                high=high_price.quantize(Decimal("0.000000000000000001")),
                low=low_price.quantize(Decimal("0.000000000000000001")),
                close=close_price.quantize(Decimal("0.000000000000000001")),
                volume=volume.quantize(Decimal("0.000000000000000001")),
                exchange="binance",
                timestamp=current_time + timedelta(hours=i),
                bid_price=(close_price - Decimal("0.50")).quantize(Decimal("0.000000000000000001")),
                ask_price=(close_price + Decimal("0.50")).quantize(Decimal("0.000000000000000001")),
            )

            data.append(market_data)

        return data

    def generate_breakout_data(
        self,
        symbol: str = "BTC/USDT",
        consolidation_periods: int = 50,
        breakout_periods: int = 20,
        breakout_strength: float = 0.05,  # 5% breakout move
        direction: int = 1,  # 1 for up, -1 for down
    ) -> List[MarketData]:
        """
        Generate breakout pattern: consolidation followed by breakout.

        Args:
            symbol: Trading symbol
            consolidation_periods: Periods of consolidation
            breakout_periods: Periods of breakout move
            breakout_strength: Strength of breakout as percentage
            direction: 1 for upward breakout, -1 for downward

        Returns:
            List of MarketData with consolidation + breakout pattern
        """
        # Generate consolidation phase
        consolidation_data = self.generate_consolidating_data(
            symbol=symbol,
            periods=consolidation_periods,
            range_pct=0.015  # Tight 1.5% range
        )

        # Get the breakout level (upper or lower bound of consolidation)
        consolidation_prices = [data.close for data in consolidation_data]
        if direction == 1:
            breakout_level = max(consolidation_prices)
        else:
            breakout_level = min(consolidation_prices)

        # Generate breakout phase
        breakout_data = []
        current_price = consolidation_data[-1].close
        current_time = consolidation_data[-1].timestamp

        for i in range(breakout_periods):
            # Strong directional move with decreasing momentum
            momentum_factor = 1.0 - (i / breakout_periods) * 0.7  # Decay momentum
            trend_strength = breakout_strength / breakout_periods * momentum_factor

            price_change = current_price * Decimal(str(direction * trend_strength))

            # Add some noise
            noise = current_price * Decimal(str(random.uniform(-0.002, 0.002)))

            open_price = current_price
            close_price = current_price + price_change + noise

            # Higher volatility during breakout
            volatility_range = current_price * Decimal("0.01")  # 1% range

            high_offset = volatility_range * Decimal(str(random.uniform(0.2, 0.6)))
            low_offset = volatility_range * Decimal(str(random.uniform(0.2, 0.6)))

            high_price = max(open_price, close_price) + high_offset
            low_price = min(open_price, close_price) - low_offset

            # High volume during breakout
            volume_multiplier = Decimal("2.0") + Decimal(str(random.uniform(0, 1)))
            volume = self.base_volume * volume_multiplier

            market_data = MarketData(
                symbol=symbol,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                exchange="binance",
                timestamp=current_time + timedelta(hours=i+1),
                bid_price=close_price - Decimal("0.50"),
                ask_price=close_price + Decimal("0.50"),
            )

            breakout_data.append(market_data)
            current_price = close_price

        return consolidation_data + breakout_data

    def generate_rsi_test_data(
        self,
        symbol: str = "BTC/USDT",
        periods: int = 100,
        target_rsi: float = 70.0,  # Target RSI level to reach
    ) -> List[MarketData]:
        """
        Generate market data designed to test RSI calculations.

        Creates a pattern that will result in a specific RSI value
        for validation of RSI calculation accuracy.

        Args:
            symbol: Trading symbol
            periods: Number of periods (need at least 14 for RSI)
            target_rsi: Target RSI value to achieve

        Returns:
            List of MarketData designed for RSI testing
        """
        if periods < 14:
            raise ValueError("Need at least 14 periods for RSI calculation")

        data = []
        current_price = self.base_price
        current_time = datetime.now(timezone.utc) - timedelta(hours=periods)

        # Create initial data points
        for i in range(14):
            # Alternating small moves to establish baseline
            direction = 1 if i % 2 == 0 else -1
            price_change = current_price * Decimal(str(direction * 0.001))  # 0.1% moves

            open_price = current_price
            close_price = current_price + price_change

            # Minimal OHLC range
            hl_range = current_price * Decimal("0.0005")
            high_price = max(open_price, close_price) + hl_range
            low_price = min(open_price, close_price) - hl_range

            volume = self.base_volume

            # Quantize all prices to 18 decimal places for crypto precision
            market_data = MarketData(
                symbol=symbol,
                open=open_price.quantize(Decimal("0.000000000000000001")),
                high=high_price.quantize(Decimal("0.000000000000000001")),
                low=low_price.quantize(Decimal("0.000000000000000001")),
                close=close_price.quantize(Decimal("0.000000000000000001")),
                volume=volume.quantize(Decimal("0.000000000000000001")),
                exchange="binance",
                timestamp=current_time + timedelta(hours=i),
                bid_price=(close_price - Decimal("0.50")).quantize(Decimal("0.000000000000000001")),
                ask_price=(close_price + Decimal("0.50")).quantize(Decimal("0.000000000000000001")),
            )

            data.append(market_data)
            current_price = close_price

        # Generate remaining periods to target specific RSI
        for i in range(14, periods):
            # Determine direction needed to reach target RSI
            if target_rsi > 70:
                # Need more up moves
                direction = 1
                strength = 0.01  # 1% moves
            elif target_rsi < 30:
                # Need more down moves
                direction = -1
                strength = 0.01  # 1% moves
            else:
                # Balanced moves for neutral RSI
                direction = 1 if random.random() > 0.5 else -1
                strength = 0.005  # 0.5% moves

            price_change = current_price * Decimal(str(direction * strength))

            open_price = current_price
            close_price = current_price + price_change

            hl_range = current_price * Decimal("0.002")
            high_price = max(open_price, close_price) + hl_range * Decimal(str(random.random()))
            low_price = min(open_price, close_price) - hl_range * Decimal(str(random.random()))

            volume = self.base_volume * Decimal(str(random.uniform(0.8, 1.2)))

            # Quantize all prices to 18 decimal places for crypto precision
            market_data = MarketData(
                symbol=symbol,
                open=open_price.quantize(Decimal("0.000000000000000001")),
                high=high_price.quantize(Decimal("0.000000000000000001")),
                low=low_price.quantize(Decimal("0.000000000000000001")),
                close=close_price.quantize(Decimal("0.000000000000000001")),
                volume=volume.quantize(Decimal("0.000000000000000001")),
                exchange="binance",
                timestamp=current_time + timedelta(hours=i),
                bid_price=(close_price - Decimal("0.50")).quantize(Decimal("0.000000000000000001")),
                ask_price=(close_price + Decimal("0.50")).quantize(Decimal("0.000000000000000001")),
            )

            data.append(market_data)
            current_price = close_price

        return data


def create_test_market_data_suite() -> dict[str, List[MarketData]]:
    """
    Create a comprehensive suite of test market data for various scenarios.

    Returns:
        Dictionary of market data lists for different testing scenarios
    """
    generator = MarketDataGenerator(seed=42)  # Deterministic for tests

    suite = {
        "uptrend_strong": generator.generate_trending_data(
            periods=100, trend_strength=0.002, direction=1
        ),
        "downtrend_strong": generator.generate_trending_data(
            periods=100, trend_strength=0.002, direction=-1
        ),
        "uptrend_weak": generator.generate_trending_data(
            periods=100, trend_strength=0.0005, direction=1
        ),
        "consolidation_tight": generator.generate_consolidating_data(
            periods=100, range_pct=0.01
        ),
        "consolidation_wide": generator.generate_consolidating_data(
            periods=100, range_pct=0.03
        ),
        "breakout_up": generator.generate_breakout_data(
            consolidation_periods=50, breakout_periods=20, direction=1
        ),
        "breakout_down": generator.generate_breakout_data(
            consolidation_periods=50, breakout_periods=20, direction=-1
        ),
        "rsi_overbought": generator.generate_rsi_test_data(
            periods=50, target_rsi=75.0
        ),
        "rsi_oversold": generator.generate_rsi_test_data(
            periods=50, target_rsi=25.0
        ),
        "rsi_neutral": generator.generate_rsi_test_data(
            periods=50, target_rsi=50.0
        ),
    }

    return suite