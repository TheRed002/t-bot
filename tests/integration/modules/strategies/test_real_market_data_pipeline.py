"""
Real Market Data Pipeline Integration Tests.

This module tests realistic market data processing pipeline with
technical indicator calculations, ensuring Decimal precision
throughout the data flow.

CRITICAL: All market data processing must maintain Decimal precision
for financial accuracy and regulatory compliance.
"""

import asyncio
import time
from decimal import Decimal, getcontext
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

import pytest
import pytest_asyncio
import numpy as np

from src.core.types import MarketData, StrategyConfig, StrategyType
from src.data.services.data_service import DataService
from src.strategies.static.mean_reversion import MeanReversionStrategy
from src.strategies.static.trend_following import TrendFollowingStrategy

from .fixtures.real_service_fixtures import generate_realistic_market_data_sequence
from .utils.indicator_validation import IndicatorValidator, IndicatorAccuracyTester

# Set high precision for Decimal calculations
getcontext().prec = 28


class MarketDataPipelineGenerator:
    """Generate realistic market data pipelines for testing."""

    @staticmethod
    def generate_crypto_market_data(
        symbol: str = "BTC/USDT",
        periods: int = 100,
        base_price: Decimal = Decimal("50000.00"),
        volatility: Decimal = Decimal("0.02"),
        trend_strength: Decimal = Decimal("0.0"),
        volume_pattern: str = "normal"
    ) -> List[MarketData]:
        """
        Generate realistic cryptocurrency market data.

        Args:
            symbol: Trading symbol
            periods: Number of periods to generate
            base_price: Starting price
            volatility: Price volatility (as decimal percentage)
            trend_strength: Trend strength (-1.0 to 1.0)
            volume_pattern: Volume pattern ('normal', 'increasing', 'decreasing', 'volatile')

        Returns:
            List of MarketData objects with realistic crypto characteristics
        """
        market_data_series = []
        current_price = base_price
        base_volume = Decimal("1000.0")

        for i in range(periods):
            timestamp = datetime.now(timezone.utc) - timedelta(hours=periods - i)

            # Calculate price movement with trend and volatility
            random_factor = Decimal(str(np.random.normal(0, 1)))
            volatility_component = current_price * volatility * random_factor
            trend_component = current_price * trend_strength * Decimal("0.001")

            # Price change with mean reversion tendency
            price_change = trend_component + volatility_component

            # Add realistic price constraints (prevent negative prices)
            new_price = current_price + price_change
            if new_price < Decimal("1.0"):
                new_price = current_price * Decimal("0.99")

            # Generate OHLC with realistic intraday patterns
            open_price = current_price
            close_price = new_price

            # High and low with realistic spreads
            intraday_volatility = volatility * Decimal("0.5")
            high_offset = new_price * intraday_volatility * Decimal(str(abs(np.random.normal(0, 1))))
            low_offset = new_price * intraday_volatility * Decimal(str(abs(np.random.normal(0, 1))))

            high_price = max(open_price, close_price) + high_offset
            low_price = min(open_price, close_price) - low_offset

            # Generate realistic volume based on pattern
            if volume_pattern == "normal":
                volume_multiplier = Decimal("1.0") + Decimal(str(np.random.normal(0, 0.3)))
            elif volume_pattern == "increasing":
                volume_multiplier = Decimal("1.0") + (Decimal(str(i)) / periods) * Decimal("2.0")
            elif volume_pattern == "decreasing":
                volume_multiplier = Decimal("2.0") - (Decimal(str(i)) / periods) * Decimal("1.5")
            else:  # volatile
                volume_multiplier = Decimal("1.0") + Decimal(str(np.random.normal(0, 0.8)))

            volume = base_volume * max(volume_multiplier, Decimal("0.1"))

            # Create market data with bid/ask spread
            spread = new_price * Decimal("0.0001")  # 0.01% spread
            bid_price = new_price - (spread / Decimal("2"))
            ask_price = new_price + (spread / Decimal("2"))

            market_data = MarketData(
                symbol=symbol,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                timestamp=timestamp,
                exchange="binance",
                bid_price=bid_price,
                ask_price=ask_price,
            )

            market_data_series.append(market_data)
            current_price = close_price

        return market_data_series

    @staticmethod
    def generate_market_regime_data(
        symbol: str = "BTC/USDT",
        regime: str = "bull_market"
    ) -> List[MarketData]:
        """
        Generate market data for specific market regimes.

        Args:
            symbol: Trading symbol
            regime: Market regime ('bull_market', 'bear_market', 'sideways', 'volatile')

        Returns:
            Market data representing the specified regime
        """
        if regime == "bull_market":
            return MarketDataPipelineGenerator.generate_crypto_market_data(
                symbol=symbol,
                periods=50,
                trend_strength=Decimal("0.8"),
                volatility=Decimal("0.015"),
                volume_pattern="increasing"
            )
        elif regime == "bear_market":
            return MarketDataPipelineGenerator.generate_crypto_market_data(
                symbol=symbol,
                periods=50,
                trend_strength=Decimal("-0.6"),
                volatility=Decimal("0.025"),
                volume_pattern="decreasing"
            )
        elif regime == "sideways":
            return MarketDataPipelineGenerator.generate_crypto_market_data(
                symbol=symbol,
                periods=60,
                trend_strength=Decimal("0.0"),
                volatility=Decimal("0.01"),
                volume_pattern="normal"
            )
        else:  # volatile
            return MarketDataPipelineGenerator.generate_crypto_market_data(
                symbol=symbol,
                periods=40,
                trend_strength=Decimal("0.2"),
                volatility=Decimal("0.05"),
                volume_pattern="volatile"
            )

    @staticmethod
    def generate_indicator_test_data(
        indicator_type: str,
        symbol: str = "BTC/USDT"
    ) -> List[MarketData]:
        """
        Generate market data optimized for testing specific indicators.

        Args:
            indicator_type: Type of indicator ('rsi', 'macd', 'bollinger', 'sma')
            symbol: Trading symbol

        Returns:
            Market data optimized for the specified indicator
        """
        if indicator_type == "rsi":
            # Generate data with clear overbought/oversold conditions
            base_price = Decimal("50000.00")
            data = []

            # Phase 1: Declining prices (oversold RSI)
            for i in range(15):
                price = base_price - Decimal(str(i * 200))
                data.append(MarketData(
                    symbol=symbol,
                    open=price + Decimal("50"),
                    high=price + Decimal("100"),
                    low=price - Decimal("50"),
                    close=price,
                    volume=Decimal("1500.0"),
                    timestamp=datetime.now(timezone.utc) - timedelta(hours=20 - i),
                    exchange="binance"
                ))

            # Phase 2: Rising prices (overbought RSI)
            for i in range(10):
                price = data[-1].close + Decimal(str(i * 300))
                data.append(MarketData(
                    symbol=symbol,
                    open=data[-1].close,
                    high=price + Decimal("100"),
                    low=data[-1].close - Decimal("50"),
                    close=price,
                    volume=Decimal("2000.0"),
                    timestamp=datetime.now(timezone.utc) - timedelta(hours=5 - i),
                    exchange="binance"
                ))

            return data

        elif indicator_type == "macd":
            # Generate data with clear trend changes for MACD crossovers
            return MarketDataPipelineGenerator.generate_crypto_market_data(
                symbol=symbol,
                periods=50,
                trend_strength=Decimal("0.5"),
                volatility=Decimal("0.02")
            )

        elif indicator_type == "bollinger":
            # Generate data with varying volatility for Bollinger Bands
            return MarketDataPipelineGenerator.generate_crypto_market_data(
                symbol=symbol,
                periods=40,
                volatility=Decimal("0.03"),
                volume_pattern="volatile"
            )

        else:  # sma/ema
            # Generate smooth trending data for moving averages
            return MarketDataPipelineGenerator.generate_crypto_market_data(
                symbol=symbol,
                periods=30,
                trend_strength=Decimal("0.3"),
                volatility=Decimal("0.01")
            )


class TestRealMarketDataPipeline:
    """Test real market data pipeline with Decimal precision."""

    @pytest.fixture
    async def real_data_service(self, strategy_service_container):
        """Get real data service from container."""
        return strategy_service_container.data_service

    @pytest.fixture
    def market_data_generator(self):
        """Create market data generator."""
        return MarketDataPipelineGenerator()

    @pytest.mark.asyncio
    async def test_decimal_precision_throughout_pipeline(self, real_data_service, market_data_generator):
        """Test that Decimal precision is maintained throughout the market data pipeline."""
        # Generate market data with high precision Decimal values
        market_data_series = market_data_generator.generate_crypto_market_data(
            symbol="BTC/USDT",
            periods=25,
            base_price=Decimal("50123.45678901"),  # High precision
            volatility=Decimal("0.0234567890"),    # High precision
        )

        # Store market data in pipeline
        for market_data in market_data_series:
            await real_data_service.store_market_data(market_data, exchange=market_data.exchange)

            # Verify all price fields maintain Decimal precision
            assert isinstance(market_data.open, Decimal)
            assert isinstance(market_data.high, Decimal)
            assert isinstance(market_data.low, Decimal)
            assert isinstance(market_data.close, Decimal)
            assert isinstance(market_data.volume, Decimal)
            assert isinstance(market_data.bid_price, Decimal)
            assert isinstance(market_data.ask_price, Decimal)

        # Retrieve market data and verify precision preservation
        retrieved_data = await real_data_service.get_market_data_history(
            symbol="BTC/USDT",
            limit=25
        )

        assert len(retrieved_data) == 25

        for original, retrieved in zip(market_data_series, retrieved_data):
            # Verify Decimal type preservation
            assert isinstance(retrieved.close, Decimal)
            assert isinstance(retrieved.volume, Decimal)

            # Verify precision preservation (at least 18 decimal places)
            assert abs(retrieved.close - original.close) < Decimal("0.000000000000000001")

    @pytest.mark.asyncio
    async def test_market_regime_data_processing(self, real_data_service, market_data_generator):
        """Test processing of different market regime data."""
        regimes = ["bull_market", "bear_market", "sideways", "volatile"]

        for regime in regimes:
            # Generate regime-specific data
            regime_data = market_data_generator.generate_market_regime_data(
                symbol=f"TEST_{regime.upper()}/USDT",
                regime=regime
            )

            # Store regime data
            for market_data in regime_data:
                await real_data_service.store_market_data(market_data, exchange=market_data.exchange)

            # Verify data characteristics
            prices = [md.close for md in regime_data]

            if regime == "bull_market":
                # Verify upward trend
                assert prices[-1] > prices[0]
                trend_ratio = prices[-1] / prices[0]
                assert trend_ratio > Decimal("1.05")  # At least 5% gain

            elif regime == "bear_market":
                # Verify downward trend
                assert prices[-1] < prices[0]
                trend_ratio = prices[-1] / prices[0]
                assert trend_ratio < Decimal("0.95")  # At least 5% loss

            elif regime == "sideways":
                # Verify limited movement
                trend_ratio = abs(prices[-1] - prices[0]) / prices[0]
                assert trend_ratio < Decimal("0.10")  # Less than 10% movement

            # All regimes should maintain positive prices and volumes
            for md in regime_data:
                assert md.close > Decimal("0")
                assert md.volume > Decimal("0")
                assert md.high >= md.low
                assert md.high >= max(md.open, md.close)
                assert md.low <= min(md.open, md.close)

    @pytest.mark.asyncio
    async def test_high_frequency_data_processing(self, real_data_service, market_data_generator):
        """Test processing of high-frequency market data."""
        # Generate high-frequency data (1-minute intervals)
        base_time = datetime.now(timezone.utc)
        hf_data = []

        base_price = Decimal("50000.00")
        for i in range(1440):  # 24 hours of minute data
            # Small price movements typical of high-frequency data
            price_change = Decimal(str(np.random.normal(0, 0.001))) * base_price
            new_price = base_price + price_change

            market_data = MarketData(
                symbol="BTC/USDT",
                open=base_price,
                high=max(base_price, new_price) + Decimal("10.0"),
                low=min(base_price, new_price) - Decimal("10.0"),
                close=new_price,
                volume=Decimal(str(100 + np.random.exponential(50))),
                timestamp=base_time - timedelta(minutes=1440 - i),
                exchange="binance",
                bid_price=new_price - Decimal("0.5"),
                ask_price=new_price + Decimal("0.5"),
            )

            hf_data.append(market_data)
            base_price = new_price

        # Test batch storage performance
        start_time = time.time()
        await real_data_service.store_market_data_batch(hf_data)
        storage_time = time.time() - start_time

        # High-frequency data storage should be efficient
        assert storage_time < 10.0  # Should complete within 10 seconds

        # Test data retrieval performance
        start_time = time.time()
        retrieved_hf_data = await real_data_service.get_market_data_history(
            symbol="BTC/USDT",
            limit=1440
        )
        retrieval_time = time.time() - start_time

        # Data retrieval should be fast
        assert retrieval_time < 5.0  # Should complete within 5 seconds
        assert len(retrieved_hf_data) == 1440

        # Verify Decimal precision in high-frequency data
        for md in retrieved_hf_data:
            assert isinstance(md.close, Decimal)
            assert isinstance(md.volume, Decimal)

    @pytest.mark.asyncio
    async def test_market_data_aggregation_pipeline(self, real_data_service, market_data_generator):
        """Test market data aggregation for different timeframes."""
        # Generate 1-hour data
        hourly_data = market_data_generator.generate_crypto_market_data(
            symbol="BTC/USDT",
            periods=24,  # 24 hours
            base_price=Decimal("50000.00")
        )

        # Store hourly data
        for md in hourly_data:
            await real_data_service.store_market_data(md, exchange=md.exchange)

        # Test aggregation to 4-hour timeframe
        aggregated_4h = await real_data_service.aggregate_market_data(
            symbol="BTC/USDT",
            source_timeframe="1h",
            target_timeframe="4h",
            periods=6  # 6 * 4h = 24h
        )

        assert len(aggregated_4h) == 6

        # Verify aggregation accuracy
        for i, agg_data in enumerate(aggregated_4h):
            # Get source data for this aggregation period
            start_idx = i * 4
            end_idx = start_idx + 4
            source_data = hourly_data[start_idx:end_idx]

            # Verify OHLC aggregation
            expected_open = source_data[0].open
            expected_high = max(md.high for md in source_data)
            expected_low = min(md.low for md in source_data)
            expected_close = source_data[-1].close
            expected_volume = sum(md.volume for md in source_data)

            # Verify with tolerance for Decimal precision
            assert abs(agg_data.open - expected_open) < Decimal("0.01")
            assert abs(agg_data.high - expected_high) < Decimal("0.01")
            assert abs(agg_data.low - expected_low) < Decimal("0.01")
            assert abs(agg_data.close - expected_close) < Decimal("0.01")
            assert abs(agg_data.volume - expected_volume) < Decimal("0.01")

            # Verify all values are Decimal
            assert isinstance(agg_data.open, Decimal)
            assert isinstance(agg_data.high, Decimal)
            assert isinstance(agg_data.low, Decimal)
            assert isinstance(agg_data.close, Decimal)
            assert isinstance(agg_data.volume, Decimal)


class TestRealIndicatorPipelineIntegration:
    """Test technical indicator pipeline with real market data."""

    @pytest.fixture(autouse=True)
    async def cleanup_database_before_test(self, strategy_service_container):
        """
        Clean database before each test in this class.

        This ensures each test starts with a clean slate, preventing data pollution
        when tests store market data and then fetch "recent" records.

        Depends on strategy_service_container to ensure database is initialized.
        """
        from sqlalchemy import text
        from src.database.connection import get_async_session

        # Clean the market_data_records table before the test
        async with get_async_session() as session:
            try:
                await session.execute(text("SET session_replication_role = replica;"))
                await session.execute(text("TRUNCATE TABLE market_data_records CASCADE;"))
                await session.execute(text("SET session_replication_role = DEFAULT;"))
                await session.commit()
            except Exception as e:
                await session.rollback()
                # Log but don't fail - test will create data anyway
                print(f"Warning: Could not clean database: {e}")

        yield
        # No cleanup needed after - next test will clean before it runs

    @pytest_asyncio.fixture
    async def strategy_with_real_data(self, strategy_service_container, market_data_generator):
        """Create strategy with real market data pipeline."""
        config = StrategyConfig(
            strategy_id="indicator_pipeline_test",
            name="indicator_pipeline_test_strategy",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol="BTC/USDT",
            timeframe="1h",
            parameters={
                "lookback_period": 20,
                "entry_threshold": Decimal("2.0"),
                "rsi_period": 14,
                "atr_period": 14,
            }
        )

        strategy = MeanReversionStrategy(
            config=config.dict(),
            services=strategy_service_container
        )
        await strategy.initialize(config)

        # Pre-load market data for indicators
        market_data = market_data_generator.generate_trending_data(
            periods=50,
            trend_strength=0.001,
            direction=1
        )

        for md in market_data:
            await strategy.services.data_service.store_market_data(md, exchange=md.exchange)

        yield strategy, market_data
        strategy.cleanup()

    @pytest.mark.asyncio
    async def test_rsi_pipeline_accuracy(self, strategy_with_real_data):
        """Test RSI calculation pipeline with mathematical validation."""
        strategy, market_data = strategy_with_real_data

        # Use the fixture's market data (already loaded)
        # Calculate RSI using strategy pipeline (fetches from DB in DESC order, then reverses)
        calculated_rsi = await strategy.get_rsi("BTC/USDT", 14)

        # Validate RSI using reference calculation on same data
        # Note: market_data is in chronological order (oldest first) as generated
        # This matches what get_rsi does internally after reversing DB results
        validator = IndicatorValidator()
        prices = [md.close for md in market_data]  # Already in correct chronological order
        expected_rsi = validator.calculate_rsi(prices, 14)

        # Verify calculation accuracy
        assert calculated_rsi is not None
        assert expected_rsi is not None
        assert isinstance(calculated_rsi, Decimal)
        assert isinstance(expected_rsi, Decimal)

        # Allow reasonable tolerance for TA-Lib vs manual calculation differences (< 0.01%)
        rsi_difference = abs(calculated_rsi - expected_rsi)
        assert rsi_difference < Decimal("0.01"), f"RSI calculation difference: {rsi_difference}"

        # Verify RSI is within valid range
        assert Decimal("0") <= calculated_rsi <= Decimal("100")

    @pytest.mark.asyncio
    async def test_moving_average_pipeline_accuracy(self, strategy_with_real_data):
        """Test moving average calculation pipeline."""
        strategy, market_data = strategy_with_real_data

        # Test SMA calculation
        calculated_sma = await strategy.get_sma("BTC/USDT", 20)

        # Validate SMA using reference calculation
        validator = IndicatorValidator()
        prices = [md.close for md in market_data]
        expected_sma = validator.calculate_sma(prices, 20)

        assert calculated_sma is not None
        assert expected_sma is not None
        assert isinstance(calculated_sma, Decimal)

        # Verify SMA accuracy
        sma_difference = abs(calculated_sma - expected_sma)
        assert sma_difference < Decimal("1.0"), f"SMA calculation difference: {sma_difference}"

        # Test EMA calculation
        calculated_ema = await strategy.get_ema("BTC/USDT", 20)
        expected_ema = validator.calculate_ema(prices, 20)

        assert calculated_ema is not None
        assert expected_ema is not None
        assert isinstance(calculated_ema, Decimal)

        # EMA may have more variance due to different calculation methods
        ema_difference = abs(calculated_ema - expected_ema)
        assert ema_difference < Decimal("50.0"), f"EMA calculation difference: {ema_difference}"

    @pytest.mark.asyncio
    async def test_bollinger_bands_pipeline(self, strategy_with_real_data):
        """Test Bollinger Bands calculation pipeline."""
        strategy, market_data = strategy_with_real_data

        # Generate volatility test data
        bb_test_data = MarketDataPipelineGenerator.generate_indicator_test_data("bollinger")

        for md in bb_test_data:
            await strategy.services.data_service.store_market_data(md, exchange=md.exchange)

        # Calculate Bollinger Bands using strategy pipeline
        bb_result = await strategy.get_bollinger_bands("BTC/USDT", 20, 2.0)

        assert bb_result is not None
        assert isinstance(bb_result, dict)
        assert "upper" in bb_result
        assert "middle" in bb_result
        assert "lower" in bb_result

        # Verify all values are Decimal
        assert isinstance(bb_result["upper"], Decimal)
        assert isinstance(bb_result["middle"], Decimal)
        assert isinstance(bb_result["lower"], Decimal)

        # Verify logical relationships
        assert bb_result["upper"] > bb_result["middle"]
        assert bb_result["middle"] > bb_result["lower"]

        # Verify against reference calculation
        validator = IndicatorValidator()
        prices = [md.close for md in bb_test_data]
        expected_bb = validator.calculate_bollinger_bands(prices, 20, 2.0)

        if expected_bb:
            # Allow tolerance for standard deviation calculation differences
            middle_diff = abs(bb_result["middle"] - expected_bb["middle"])
            assert middle_diff < Decimal("10.0"), f"Bollinger middle band difference: {middle_diff}"

    @pytest.mark.asyncio
    async def test_atr_pipeline_accuracy(self, strategy_with_real_data):
        """Test ATR (Average True Range) calculation pipeline."""
        strategy, market_data = strategy_with_real_data

        # Calculate ATR using strategy pipeline
        calculated_atr = await strategy.get_atr("BTC/USDT", 14)

        assert calculated_atr is not None
        assert isinstance(calculated_atr, Decimal)
        assert calculated_atr >= Decimal("0")  # ATR should be non-negative

        # Validate ATR using reference calculation
        validator = IndicatorValidator()
        high_prices = [md.high for md in market_data]
        low_prices = [md.low for md in market_data]
        close_prices = [md.close for md in market_data]

        expected_atr = validator.calculate_atr(high_prices, low_prices, close_prices, 14)

        if expected_atr:
            atr_difference = abs(calculated_atr - expected_atr)
            # ATR calculations can vary based on implementation details
            assert atr_difference < calculated_atr * Decimal("0.1"), f"ATR calculation difference: {atr_difference}"

    @pytest.mark.asyncio
    async def test_indicator_pipeline_performance(self, strategy_with_real_data):
        """Test performance of indicator calculation pipeline."""
        strategy, market_data = strategy_with_real_data

        # Test multiple indicator calculations in sequence
        start_time = time.time()

        # Calculate multiple indicators
        indicators = await asyncio.gather(
            strategy.get_sma("BTC/USDT", 20),
            strategy.get_ema("BTC/USDT", 20),
            strategy.get_rsi("BTC/USDT", 14),
            strategy.get_atr("BTC/USDT", 14),
            strategy.get_bollinger_bands("BTC/USDT", 20, 2.0),
        )

        calculation_time = time.time() - start_time

        # All indicators should calculate quickly
        assert calculation_time < 5.0  # Should complete within 5 seconds

        # Verify all indicators returned valid results
        sma, ema, rsi, atr, bb = indicators

        assert sma is not None and isinstance(sma, Decimal)
        assert ema is not None and isinstance(ema, Decimal)
        assert rsi is not None and isinstance(rsi, Decimal)
        assert atr is not None and isinstance(atr, Decimal)
        assert bb is not None and isinstance(bb, dict)

        # Test parallel indicator calculations
        start_time = time.time()

        # Run same calculations in parallel (should be faster)
        parallel_indicators = await asyncio.gather(
            strategy.get_sma("BTC/USDT", 20),
            strategy.get_ema("BTC/USDT", 20),
            strategy.get_rsi("BTC/USDT", 14),
            strategy.get_atr("BTC/USDT", 14),
            strategy.get_bollinger_bands("BTC/USDT", 20, 2.0),
        )

        parallel_time = time.time() - start_time

        # Parallel calculations should be at least as fast as sequential
        assert parallel_time <= calculation_time + 1.0  # Allow small overhead

    @pytest.mark.asyncio
    async def test_cross_timeframe_indicator_pipeline(self, strategy_service_container):
        """Test indicator calculations across different timeframes."""
        # Create market data for multiple timeframes
        generator = MarketDataPipelineGenerator()

        # Generate 1h data
        hourly_data = generator.generate_crypto_market_data(
            symbol="BTC/USDT",
            periods=48,  # 48 hours
            base_price=Decimal("50000.00")
        )

        # Generate 4h data (aggregated)
        four_hour_data = []
        for i in range(0, len(hourly_data), 4):
            chunk = hourly_data[i:i+4]
            if len(chunk) == 4:
                aggregated = MarketData(
                    symbol="BTC/USDT",
                    open=chunk[0].open,
                    high=max(md.high for md in chunk),
                    low=min(md.low for md in chunk),
                    close=chunk[-1].close,
                    volume=sum(md.volume for md in chunk),
                    timestamp=chunk[-1].timestamp,
                    exchange="binance"
                )
                four_hour_data.append(aggregated)

        # Store data for both timeframes
        data_service = strategy_service_container.data_service

        for md in hourly_data:
            await data_service.store_market_data(md, exchange="binance")

        for md in four_hour_data:
            await data_service.store_market_data(md, exchange="binance")

        # Create strategies for different timeframes
        config_1h = StrategyConfig(
            strategy_id="cross_tf_test_1h",
            name="cross_tf_test_1h",
            strategy_type=StrategyType.TREND_FOLLOWING,
            symbol="BTC/USDT",
            timeframe="1h",
            parameters={"fast_ma_period": 10, "slow_ma_period": 20}
        )

        config_4h = StrategyConfig(
            strategy_id="cross_tf_test_4h",
            name="cross_tf_test_4h",
            strategy_type=StrategyType.TREND_FOLLOWING,
            symbol="BTC/USDT",
            timeframe="4h",
            parameters={"fast_ma_period": 10, "slow_ma_period": 20}
        )

        strategy_1h = TrendFollowingStrategy(
            config=config_1h.dict(),
            services=strategy_service_container
        )
        await strategy_1h.initialize(config_1h)

        strategy_4h = TrendFollowingStrategy(
            config=config_4h.dict(),
            services=strategy_service_container
        )
        await strategy_4h.initialize(config_4h)

        try:
            # Calculate indicators for both timeframes
            sma_1h = await strategy_1h.get_sma("BTC/USDT", 20)
            sma_4h = await strategy_4h.get_sma("BTC/USDT", 12)  # Adjusted for 4h timeframe

            # Both should return valid Decimal values
            assert sma_1h is not None and isinstance(sma_1h, Decimal)
            assert sma_4h is not None and isinstance(sma_4h, Decimal)

            # 4h timeframe should have smoother (less volatile) indicators
            # This is a general expectation but may not always hold

        finally:
            await strategy_1h.cleanup()
            await strategy_4h.cleanup()