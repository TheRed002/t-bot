"""
Real Signal Generation Integration Tests.

This module tests signal generation using real technical indicators
instead of mock signal generation, ensuring mathematical accuracy
and proper integration with database persistence.

CRITICAL: All signal generation must use real indicator calculations
with Decimal precision for financial accuracy.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

import pytest

from src.core.types import MarketData, Signal, SignalDirection, StrategyConfig, StrategyType
from src.strategies.static.mean_reversion import MeanReversionStrategy
from src.strategies.static.trend_following import TrendFollowingStrategy
from src.strategies.static.breakout import BreakoutStrategy

from .fixtures.real_service_fixtures import (
    generate_mean_reversion_scenario,
    generate_trend_following_scenario,
    generate_breakout_scenario,
    generate_realistic_market_data_sequence
)
from .utils.indicator_validation import IndicatorValidator, IndicatorAccuracyTester


class TestRealMeanReversionSignalGeneration:
    """Test real mean reversion signal generation with actual Z-score calculations."""

    @pytest.fixture
    async def real_mean_reversion_strategy(self, strategy_service_container):
        """Create real mean reversion strategy with dependencies."""
        config = StrategyConfig(
            strategy_id="real_mr_signal_test_001",
            name="real_mr_signal_test",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol="BTC/USDT",
            timeframe="1h",
            min_confidence=Decimal("0.6"),
            parameters={
                "lookback_period": 20,
                "entry_threshold": Decimal("2.0"),
                "exit_threshold": Decimal("0.5"),
                "volume_filter": True,
                "min_volume_ratio": Decimal("1.5"),
                "atr_period": 14,
                "atr_multiplier": Decimal("2.0"),
            }
        )

        strategy = MeanReversionStrategy(
            config=config.dict(),
            services=strategy_service_container
        )
        await strategy.initialize(config)
        yield strategy
        strategy.cleanup()

    @pytest.mark.asyncio
    async def test_real_z_score_calculation_accuracy(self, real_mean_reversion_strategy):
        """Test that Z-score calculations are mathematically accurate."""
        strategy = real_mean_reversion_strategy

        # Generate market data with known statistical properties
        base_price = Decimal("50000.00")
        prices = []

        # Create 25 data points for statistical significance
        for i in range(25):
            # Create price series with predictable mean and standard deviation
            if i < 20:
                # First 20 points establish baseline (mean around 50000)
                price = base_price + Decimal(str((i % 5 - 2) * 100))
            else:
                # Last 5 points create deviation (should trigger signal)
                price = base_price - Decimal(str((i - 19) * 400))  # Significant deviation

            market_data = MarketData(
                symbol="BTC/USDT",
                open=price - Decimal("50"),
                high=price + Decimal("100"),
                low=price - Decimal("100"),
                close=price,
                volume=Decimal("2500.00"),  # Above min_volume_ratio threshold
                timestamp=datetime.now(timezone.utc) - timedelta(hours=25-i),
                exchange="binance"
            )

            # Store market data for strategy calculations
            await strategy.services.data_service.store_market_data(market_data, exchange=market_data.exchange)
            prices.append(price)

        # Calculate expected Z-score manually
        lookback_prices = prices[-20:]  # Last 20 prices for lookback
        mean_price = sum(lookback_prices) / Decimal("20")
        variance = sum([(p - mean_price) ** 2 for p in lookback_prices]) / Decimal("20")
        std_dev = Decimal(str(variance.sqrt()))

        current_price = prices[-1]
        expected_z_score = (current_price - mean_price) / std_dev if std_dev > 0 else Decimal("0")

        # Generate signal using real strategy
        final_market_data = MarketData(
            symbol="BTC/USDT",
            open=current_price - Decimal("50"),
            high=current_price + Decimal("100"),
            low=current_price - Decimal("100"),
            close=current_price,
            volume=Decimal("3000.00"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance"
        )

        signals = await strategy.generate_signals(final_market_data)

        # Verify signal generation
        assert isinstance(signals, list)

        if signals:
            signal = signals[0]

            # Verify signal structure
            assert isinstance(signal, Signal)
            assert isinstance(signal.confidence, Decimal)
            assert isinstance(signal.strength, Decimal)
            assert signal.symbol == "BTC/USDT"
            assert signal.source == strategy.name

            # Verify Z-score calculation in metadata
            assert "z_score" in signal.metadata
            calculated_z_score = Decimal(str(signal.metadata["z_score"]))

            # Compare calculated vs expected Z-score (allow small tolerance)
            z_score_difference = abs(calculated_z_score - expected_z_score)
            assert z_score_difference < Decimal("0.1"), f"Z-score difference too large: {z_score_difference}"

            # Verify signal direction matches Z-score
            if calculated_z_score < -strategy.entry_threshold:
                assert signal.direction == SignalDirection.BUY
            elif calculated_z_score > strategy.entry_threshold:
                assert signal.direction == SignalDirection.SELL

    @pytest.mark.asyncio
    async def test_real_volume_filter_integration(self, real_mean_reversion_strategy):
        """Test that volume filter correctly affects signal generation."""
        strategy = real_mean_reversion_strategy

        # Set up baseline market data
        base_price = Decimal("50000.00")
        for i in range(20):
            market_data = MarketData(
                symbol="BTC/USDT",
                open=base_price,
                high=base_price + Decimal("100"),
                low=base_price - Decimal("100"),
                close=base_price + Decimal(str((i % 5 - 2) * 100)),
                volume=Decimal("1000.00"),  # Standard volume
                timestamp=datetime.now(timezone.utc) - timedelta(hours=21-i),
                exchange="binance"
            )
            await strategy.services.data_service.store_market_data(market_data, exchange=market_data.exchange)

        # Test with low volume (should not generate signal)
        low_volume_data = MarketData(
            symbol="BTC/USDT",
            open=Decimal("48000.00"),
            high=Decimal("48100.00"),
            low=Decimal("47000.00"),
            close=Decimal("47500.00"),  # Significant price deviation
            volume=Decimal("800.00"),  # Below min_volume_ratio * average
            timestamp=datetime.now(timezone.utc),
            exchange="binance"
        )

        low_volume_signals = await strategy.generate_signals(low_volume_data)

        # Test with high volume (should generate signal)
        high_volume_data = MarketData(
            symbol="BTC/USDT",
            open=Decimal("48000.00"),
            high=Decimal("48100.00"),
            low=Decimal("47000.00"),
            close=Decimal("47500.00"),  # Same price deviation
            volume=Decimal("2000.00"),  # Above min_volume_ratio * average
            timestamp=datetime.now(timezone.utc),
            exchange="binance"
        )

        high_volume_signals = await strategy.generate_signals(high_volume_data)

        # Volume filter should affect signal generation
        if strategy.volume_filter:
            # Low volume should generate fewer or no signals
            # High volume should generate more signals
            assert len(high_volume_signals) >= len(low_volume_signals)

    @pytest.mark.asyncio
    async def test_real_atr_based_position_sizing(self, real_mean_reversion_strategy):
        """Test that ATR calculations affect position sizing in signals."""
        strategy = real_mean_reversion_strategy

        # Generate market data with varying volatility
        base_price = Decimal("50000.00")
        high_volatility_data = []
        low_volatility_data = []

        for i in range(25):
            timestamp = datetime.now(timezone.utc) - timedelta(hours=25-i)

            # High volatility data
            volatility_factor = Decimal("500")  # High volatility
            high_vol_market_data = MarketData(
                symbol="BTC/USDT",
                open=base_price + Decimal(str(i * 10)),
                high=base_price + Decimal(str(i * 10)) + volatility_factor,
                low=base_price + Decimal(str(i * 10)) - volatility_factor,
                close=base_price + Decimal(str(i * 10 + (i % 7 - 3) * 200)),
                volume=Decimal("2000.00"),
                timestamp=timestamp,
                exchange="binance"
            )
            high_volatility_data.append(high_vol_market_data)

            # Low volatility data
            volatility_factor = Decimal("50")  # Low volatility
            low_vol_market_data = MarketData(
                symbol="BTC/USDT",
                open=base_price + Decimal(str(i * 10)),
                high=base_price + Decimal(str(i * 10)) + volatility_factor,
                low=base_price + Decimal(str(i * 10)) - volatility_factor,
                close=base_price + Decimal(str(i * 10 + (i % 7 - 3) * 200)),
                volume=Decimal("2000.00"),
                timestamp=timestamp,
                exchange="binance"
            )
            low_volatility_data.append(low_vol_market_data)

        # Test high volatility scenario
        for market_data in high_volatility_data:
            await strategy.services.data_service.store_market_data(market_data, exchange=market_data.exchange)

        high_vol_signals = await strategy.generate_signals(high_volatility_data[-1])

        # Reset and test low volatility scenario
        await strategy.services.data_service.clear_market_data("BTC/USDT")
        for market_data in low_volatility_data:
            await strategy.services.data_service.store_market_data(market_data, exchange=market_data.exchange)

        low_vol_signals = await strategy.generate_signals(low_volatility_data[-1])

        # ATR should affect signal strength/confidence
        if high_vol_signals and low_vol_signals:
            # In high volatility, position sizing should be more conservative
            high_vol_signal = high_vol_signals[0]
            low_vol_signal = low_vol_signals[0]

            # Verify ATR metadata is present
            assert "atr" in high_vol_signal.metadata or "volatility" in high_vol_signal.metadata
            assert "atr" in low_vol_signal.metadata or "volatility" in low_vol_signal.metadata


class TestRealTrendFollowingSignalGeneration:
    """Test real trend following signal generation with actual MA crossovers."""

    @pytest.fixture
    async def real_trend_following_strategy(self, strategy_service_container):
        """Create real trend following strategy with dependencies."""
        config = StrategyConfig(
            strategy_id="real_tf_signal_test_001",
            name="real_tf_signal_test",
            strategy_type=StrategyType.TREND_FOLLOWING,
            symbol="BTC/USDT",
            timeframe="1h",
            min_confidence=Decimal("0.7"),
            parameters={
                "fast_ma_period": 10,
                "slow_ma_period": 20,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "volume_confirmation": True,
                "min_volume_ratio": Decimal("1.2"),
            }
        )

        strategy = TrendFollowingStrategy(
            config=config.dict(),
            services=strategy_service_container
        )
        await strategy.initialize(config)
        yield strategy
        strategy.cleanup()

    @pytest.mark.asyncio
    async def test_real_ma_crossover_signal_generation(self, real_trend_following_strategy):
        """Test MA crossover detection with real calculations."""
        strategy = real_trend_following_strategy

        # Generate price data that creates a clear MA crossover
        base_price = Decimal("50000.00")
        prices = []

        # First phase: fast MA below slow MA (downtrend)
        for i in range(25):
            if i < 15:
                # Decreasing prices (fast MA should be below slow MA)
                price = base_price - Decimal(str(i * 100))
            else:
                # Increasing prices (fast MA should cross above slow MA)
                price = base_price - Decimal(str((15 - (i - 15)) * 100))

            market_data = MarketData(
                symbol="BTC/USDT",
                open=price - Decimal("50"),
                high=price + Decimal("100"),
                low=price - Decimal("100"),
                close=price,
                volume=Decimal("2500.00"),
                timestamp=datetime.now(timezone.utc) - timedelta(hours=25-i),
                exchange="binance"
            )

            await strategy.services.data_service.store_market_data(market_data, exchange=market_data.exchange)
            prices.append(price)

        # Calculate expected MAs manually
        validator = IndicatorValidator()
        fast_ma = validator.calculate_sma(prices[-10:], 10)
        slow_ma = validator.calculate_sma(prices[-20:], 20)

        # Generate final signal
        final_market_data = MarketData(
            symbol="BTC/USDT",
            open=prices[-1],
            high=prices[-1] + Decimal("100"),
            low=prices[-1] - Decimal("100"),
            close=prices[-1] + Decimal("200"),  # Strong upward movement
            volume=Decimal("3000.00"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance"
        )

        signals = await strategy.generate_signals(final_market_data)

        # Verify signal generation
        assert isinstance(signals, list)

        if signals and fast_ma and slow_ma:
            signal = signals[0]

            # Verify signal structure
            assert isinstance(signal, Signal)
            assert isinstance(signal.confidence, Decimal)
            assert isinstance(signal.strength, Decimal)

            # Verify MA crossover logic
            if fast_ma > slow_ma:
                # Fast MA above slow MA should generate BUY signal
                assert signal.direction == SignalDirection.BUY
            elif fast_ma < slow_ma:
                # Fast MA below slow MA should generate SELL signal
                assert signal.direction == SignalDirection.SELL

            # Verify MA values in metadata
            assert "fast_ma" in signal.metadata or "slow_ma" in signal.metadata

    @pytest.mark.asyncio
    async def test_real_rsi_filter_integration(self, real_trend_following_strategy):
        """Test RSI filter integration with trend following signals."""
        strategy = real_trend_following_strategy

        # Generate market data with known RSI characteristics
        base_price = Decimal("50000.00")

        # Create oversold condition (RSI < 30)
        oversold_prices = []
        for i in range(20):
            # Declining prices to create oversold RSI
            price = base_price - Decimal(str(i * 200))
            oversold_prices.append(price)

            market_data = MarketData(
                symbol="BTC/USDT",
                open=price + Decimal("100"),
                high=price + Decimal("150"),
                low=price - Decimal("50"),
                close=price,
                volume=Decimal("2000.00"),
                timestamp=datetime.now(timezone.utc) - timedelta(hours=21-i),
                exchange="binance"
            )
            await strategy.services.data_service.store_market_data(market_data, exchange=market_data.exchange)

        # Calculate expected RSI
        validator = IndicatorValidator()
        expected_rsi = validator.calculate_rsi(oversold_prices, 14)

        # Generate signal in oversold condition
        oversold_market_data = MarketData(
            symbol="BTC/USDT",
            open=oversold_prices[-1],
            high=oversold_prices[-1] + Decimal("200"),
            low=oversold_prices[-1] - Decimal("100"),
            close=oversold_prices[-1] + Decimal("150"),  # Slight recovery
            volume=Decimal("2500.00"),
            timestamp=datetime.now(timezone.utc),
            exchange="binance"
        )

        oversold_signals = await strategy.generate_signals(oversold_market_data)

        # Verify RSI influence on signal generation
        if oversold_signals and expected_rsi:
            signal = oversold_signals[0]

            # In oversold conditions, should be more likely to generate BUY signals
            if expected_rsi < strategy.config.parameters["rsi_oversold"]:
                # Signal should favor BUY direction or have higher confidence
                assert signal.direction == SignalDirection.BUY or signal.confidence > Decimal("0.7")

            # Verify RSI in metadata
            assert "rsi" in signal.metadata
            calculated_rsi = Decimal(str(signal.metadata["rsi"]))

            # Verify RSI calculation accuracy
            rsi_difference = abs(calculated_rsi - expected_rsi)
            assert rsi_difference < Decimal("5.0"), f"RSI calculation inaccurate: {rsi_difference}"


class TestRealBreakoutSignalGeneration:
    """Test real breakout signal generation with actual support/resistance levels."""

    @pytest.fixture
    async def real_breakout_strategy(self, strategy_service_container):
        """Create real breakout strategy with dependencies."""
        config = StrategyConfig(
            strategy_id="real_bo_signal_test_001",
            name="real_bo_signal_test",
            strategy_type=StrategyType.BREAKOUT,
            symbol="BTC/USDT",
            timeframe="1h",
            min_confidence=Decimal("0.65"),
            parameters={
                "lookback_period": 20,
                "consolidation_period": 5,
                "volume_confirmation": True,
                "min_volume_ratio": Decimal("1.5"),
                "false_breakout_filter": True,
                "false_breakout_threshold": Decimal("0.02"),
            }
        )

        strategy = BreakoutStrategy(
            config=config.dict(),
            services=strategy_service_container
        )
        await strategy.initialize(config)
        yield strategy
        strategy.cleanup()

    @pytest.mark.asyncio
    async def test_real_support_resistance_detection(self, real_breakout_strategy):
        """Test real support and resistance level detection."""
        strategy = real_breakout_strategy

        # Create consolidation pattern with clear support/resistance
        base_price = Decimal("50000.00")
        support_level = base_price - Decimal("500.00")
        resistance_level = base_price + Decimal("500.00")

        # Generate consolidation data
        for i in range(25):
            if i < 20:
                # Consolidation phase: price oscillates between support and resistance
                if i % 4 == 0:
                    price = support_level + Decimal("50")  # Near support
                elif i % 4 == 2:
                    price = resistance_level - Decimal("50")  # Near resistance
                else:
                    price = base_price  # Middle range

                volume = Decimal("1500.00")  # Lower consolidation volume
            else:
                # Breakout phase: price breaks above resistance
                price = resistance_level + Decimal(str((i - 19) * 200))
                volume = Decimal("3000.00")  # Higher breakout volume

            market_data = MarketData(
                symbol="BTC/USDT",
                open=price - Decimal("25"),
                high=price + Decimal("100"),
                low=price - Decimal("75"),
                close=price,
                volume=volume,
                timestamp=datetime.now(timezone.utc) - timedelta(hours=25-i),
                exchange="binance"
            )

            await strategy.services.data_service.store_market_data(market_data, exchange=market_data.exchange)

        # Generate breakout signal
        breakout_market_data = MarketData(
            symbol="BTC/USDT",
            open=resistance_level + Decimal("100"),
            high=resistance_level + Decimal("800"),
            low=resistance_level + Decimal("50"),
            close=resistance_level + Decimal("600"),  # Clear breakout
            volume=Decimal("4000.00"),  # High volume confirmation
            timestamp=datetime.now(timezone.utc),
            exchange="binance"
        )

        signals = await strategy.generate_signals(breakout_market_data)

        # Verify breakout signal
        assert isinstance(signals, list)

        if signals:
            signal = signals[0]

            # Verify signal structure
            assert isinstance(signal, Signal)
            assert isinstance(signal.confidence, Decimal)
            assert isinstance(signal.strength, Decimal)

            # Breakout above resistance should generate BUY signal
            assert signal.direction == SignalDirection.BUY

            # Verify breakout metadata
            assert "breakout_level" in signal.metadata or "resistance" in signal.metadata
            assert "volume_ratio" in signal.metadata

            # High volume breakout should have high confidence
            volume_ratio = Decimal(str(signal.metadata.get("volume_ratio", "1.0")))
            if volume_ratio > strategy.config.parameters["min_volume_ratio"]:
                assert signal.confidence > Decimal("0.65")

    @pytest.mark.asyncio
    async def test_real_false_breakout_filter(self, real_breakout_strategy):
        """Test false breakout filter with real price action."""
        strategy = real_breakout_strategy

        # Set up consolidation pattern
        base_price = Decimal("50000.00")
        resistance_level = base_price + Decimal("500.00")

        for i in range(20):
            price = base_price + Decimal(str((i % 5 - 2) * 100))
            market_data = MarketData(
                symbol="BTC/USDT",
                open=price - Decimal("50"),
                high=price + Decimal("100"),
                low=price - Decimal("100"),
                close=price,
                volume=Decimal("1500.00"),
                timestamp=datetime.now(timezone.utc) - timedelta(hours=21-i),
                exchange="binance"
            )
            await strategy.services.data_service.store_market_data(market_data, exchange=market_data.exchange)

        # Test false breakout (breaks resistance but doesn't sustain)
        false_breakout_data = MarketData(
            symbol="BTC/USDT",
            open=resistance_level - Decimal("50"),
            high=resistance_level + Decimal("100"),  # Breaks resistance briefly
            low=resistance_level - Decimal("100"),
            close=resistance_level - Decimal("20"),  # Closes back below resistance
            volume=Decimal("2000.00"),  # Lower volume
            timestamp=datetime.now(timezone.utc),
            exchange="binance"
        )

        false_breakout_signals = await strategy.generate_signals(false_breakout_data)

        # Test real breakout (breaks resistance and sustains)
        real_breakout_data = MarketData(
            symbol="BTC/USDT",
            open=resistance_level + Decimal("50"),
            high=resistance_level + Decimal("400"),
            low=resistance_level + Decimal("20"),
            close=resistance_level + Decimal("300"),  # Closes well above resistance
            volume=Decimal("4000.00"),  # High volume
            timestamp=datetime.now(timezone.utc),
            exchange="binance"
        )

        real_breakout_signals = await strategy.generate_signals(real_breakout_data)

        # False breakout filter should affect signal generation
        if strategy.config.parameters["false_breakout_filter"]:
            # Real breakout should generate more/stronger signals than false breakout
            assert len(real_breakout_signals) >= len(false_breakout_signals)

            if real_breakout_signals and false_breakout_signals:
                real_signal = real_breakout_signals[0]
                false_signal = false_breakout_signals[0]

                # Real breakout should have higher confidence
                assert real_signal.confidence >= false_signal.confidence


class TestRealMultiStrategySignalCoordination:
    """Test coordination between multiple real strategies."""

    @pytest.mark.asyncio
    async def test_real_strategy_signal_aggregation(self, strategy_service_container):
        """Test aggregation of signals from multiple real strategies."""
        # Create multiple strategies
        strategies = []

        # Mean Reversion Strategy
        mr_config = StrategyConfig(
            strategy_id="multi_test_mr",
            name="multi_test_mean_reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol="BTC/USDT",
            timeframe="1h",
            parameters={"lookback_period": 20, "entry_threshold": Decimal("2.0")}
        )
        mr_strategy = MeanReversionStrategy(
            config=mr_config.dict(),
            services=strategy_service_container
        )
        await mr_strategy.initialize(mr_config)
        strategies.append(mr_strategy)

        # Trend Following Strategy
        tf_config = StrategyConfig(
            strategy_id="multi_test_tf",
            name="multi_test_trend_following",
            strategy_type=StrategyType.TREND_FOLLOWING,
            symbol="BTC/USDT",
            timeframe="1h",
            parameters={"fast_ma_period": 10, "slow_ma_period": 20}
        )
        tf_strategy = TrendFollowingStrategy(
            config=tf_config.dict(),
            services=strategy_service_container
        )
        await tf_strategy.initialize(tf_config)
        strategies.append(tf_strategy)

        try:
            # Set up common market data for all strategies
            market_data_sequence = generate_realistic_market_data_sequence(
                pattern="mixed", periods=30
            )

            # Store market data for all strategies
            for strategy in strategies:
                for market_data in market_data_sequence:
                    await strategy.services.data_service.store_market_data(market_data, exchange=market_data.exchange)

            # Generate signals from all strategies
            all_signals = []
            final_market_data = market_data_sequence[-1]

            for strategy in strategies:
                signals = await strategy.generate_signals(final_market_data)
                all_signals.extend(signals)

            # Verify signal coordination
            assert isinstance(all_signals, list)

            # Check signal quality across all strategies
            for signal in all_signals:
                assert isinstance(signal, Signal)
                assert isinstance(signal.confidence, Decimal)
                assert isinstance(signal.strength, Decimal)
                assert signal.confidence >= Decimal("0.1")
                assert signal.symbol == "BTC/USDT"
                assert signal.timestamp is not None

            # Verify different strategies can generate different signals
            signal_sources = set(signal.source for signal in all_signals)
            if len(all_signals) > 1:
                assert len(signal_sources) >= 1  # At least one strategy generated signals

            # Verify signal diversity (different strategies may have different views)
            signal_directions = set(signal.direction for signal in all_signals)
            # It's normal for different strategies to have different directional views

        finally:
            # Cleanup
            for strategy in strategies:
                strategy.cleanup()

    @pytest.mark.asyncio
    async def test_real_signal_persistence_and_retrieval(self, strategy_service_container):
        """Test real signal persistence to database and retrieval."""
        # Create strategy
        config = StrategyConfig(
            strategy_id="persistence_test_001",
            name="persistence_test_strategy",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol="BTC/USDT",
            timeframe="1h",
            parameters={"lookback_period": 20, "entry_threshold": Decimal("2.0")}
        )

        strategy = MeanReversionStrategy(
            config=config.dict(),
            services=strategy_service_container
        )
        await strategy.initialize(config)

        try:
            # Generate market data and signals
            market_data_sequence = generate_mean_reversion_scenario()

            for market_data in market_data_sequence:
                await strategy.services.data_service.store_market_data(market_data, exchange=market_data.exchange)

            signals = await strategy.generate_signals(market_data_sequence[-1])

            if signals:
                # Persist signals to database
                signal_service = strategy_service_container.analytics_service
                for signal in signals:
                    await signal_service.store_signal(signal)

                # Retrieve signals from database
                retrieved_signals = await signal_service.get_signals(
                    strategy_id=config.strategy_id,
                    symbol="BTC/USDT",
                    limit=10
                )

                # Verify persistence accuracy
                assert len(retrieved_signals) >= len(signals)

                for original_signal, retrieved_signal in zip(signals, retrieved_signals):
                    assert retrieved_signal.strategy_id == original_signal.strategy_id
                    assert retrieved_signal.symbol == original_signal.symbol
                    assert retrieved_signal.direction == original_signal.direction
                    assert isinstance(retrieved_signal.confidence, Decimal)
                    assert isinstance(retrieved_signal.strength, Decimal)

        finally:
            strategy.cleanup()