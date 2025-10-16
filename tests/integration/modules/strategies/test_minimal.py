"""
Minimal Strategy Integration Tests - Quick smoke tests for strategies module.

This module provides minimal integration tests to verify basic functionality
without extensive fixture overhead.
"""

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.core.types import MarketData, StrategyConfig, StrategyType
from src.strategies.static.mean_reversion import MeanReversionStrategy
from src.strategies.static.trend_following import TrendFollowingStrategy


@pytest.mark.integration
class TestMinimalStrategyIntegration:
    """Minimal strategy integration tests."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_mean_reversion_strategy_initialization(self):
        """Test mean reversion strategy can be initialized."""
        config = StrategyConfig(
            strategy_id="test_mr_001",
            name="test_mean_reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol="BTC/USDT",
            timeframe="1h",
            parameters={
                "lookback_period": 20,
                "entry_threshold": Decimal("2.0"),
                "mean_period": 20,
                "deviation_threshold": Decimal("2.0"),
                "reversion_strength": Decimal("0.5"),
            },
        )

        strategy = MeanReversionStrategy(config=config.model_dump())
        await strategy.initialize(config)

        assert strategy.name == "test_mean_reversion"
        assert strategy.config.strategy_type == StrategyType.MEAN_REVERSION

        await strategy.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_trend_following_strategy_initialization(self):
        """Test trend following strategy can be initialized."""
        config = StrategyConfig(
            strategy_id="test_tf_001",
            name="test_trend_following",
            strategy_type=StrategyType.TREND_FOLLOWING,
            symbol="BTC/USDT",
            timeframe="1h",
            parameters={
                "fast_ma_period": 10,
                "slow_ma_period": 20,
                "rsi_period": 14,
            },
        )

        strategy = TrendFollowingStrategy(config=config.model_dump())
        await strategy.initialize(config)

        assert strategy.name == "test_trend_following"
        assert strategy.config.strategy_type == StrategyType.TREND_FOLLOWING

        await strategy.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_strategy_signal_generation_no_data(self):
        """Test strategy signal generation with no historical data."""
        config = StrategyConfig(
            strategy_id="test_signals_001",
            name="test_signals",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol="BTC/USDT",
            timeframe="1h",
            parameters={
                "lookback_period": 20,
                "entry_threshold": Decimal("2.0"),
                "mean_period": 20,
                "deviation_threshold": Decimal("2.0"),
                "reversion_strength": Decimal("0.5"),
            },
        )

        strategy = MeanReversionStrategy(config=config.model_dump())
        await strategy.initialize(config)

        # Generate signals with single market data point (no history)
        market_data = MarketData(
            symbol="BTC/USDT",
            open=Decimal("50000.00"),
            high=Decimal("51000.00"),
            low=Decimal("49000.00"),
            close=Decimal("50500.00"),
            volume=Decimal("1000.00"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance",
        )

        signals = await strategy.generate_signals(market_data)

        # Should return empty list or valid signals
        assert isinstance(signals, list)
        # With no history, strategy may not generate signals

        await strategy.cleanup()
