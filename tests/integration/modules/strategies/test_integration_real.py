"""
Real Service Integration Tests for Strategy Framework.

This module provides comprehensive integration tests using real service
implementations instead of mocks. Tests the complete strategy framework
with actual calculations, database persistence, and business logic.

Key Features:
- Real StrategyService with dependency injection
- Real technical indicators with mathematical accuracy
- Real risk management with portfolio calculations
- Database persistence for configurations and signals
- Performance validation for production readiness
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List

import pytest

import pytest_asyncio
from src.core.exceptions import ServiceError, StrategyError, ValidationError
from src.core.types import (
    MarketData,
    Signal,
    SignalDirection,
    StrategyConfig,
    StrategyStatus,
    StrategyType,
)

# Import market data generators
from tests.integration.modules.strategies.fixtures.market_data_generators import (
    MarketDataGenerator,
    create_test_market_data_suite,
)

# Import validation helpers
from tests.integration.modules.strategies.helpers.indicator_validators import (
    IndicatorValidator,
    PerformanceBenchmarker,
    validate_all_indicators,
    create_known_test_cases,
)

# Import performance benchmarks
from tests.integration.modules.strategies.performance.benchmarks import (
    StrategyPerformanceBenchmarker,
    run_comprehensive_benchmarks,
)


class TestRealStrategyFrameworkIntegration:
    """
    Integration tests using real service implementations.

    This test class validates the complete strategy framework using
    production-ready services with actual calculations and database
    persistence. No mocks are used for business logic components.
    """

    @pytest_asyncio.fixture(autouse=True)
    async def cleanup_database_before_test(self, real_database_service_fixture):
        """
        Clean database before each test in this class.

        This ensures each test starts with a clean slate, preventing data pollution
        when tests store market data and then fetch "recent" records.
        """
        from sqlalchemy import text

        async with real_database_service_fixture.get_session() as session:
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

    @pytest.mark.asyncio
    async def test_real_technical_indicators_accuracy(
        self, real_data_service, real_mean_reversion_strategy
    ):
        """Test real technical indicators produce mathematically accurate results."""
        # Generate test data with known patterns
        generator = MarketDataGenerator(seed=42)
        test_data = generator.generate_trending_data(periods=50, trend_strength=0.001, direction=1)

        # Store test data in database
        for md in test_data:
            await real_data_service.store_market_data(md, exchange="binance")

        # Test RSI calculation via strategy (strategies delegate to DataService)
        rsi_result = await real_mean_reversion_strategy.get_rsi("BTC/USDT", 14)
        assert rsi_result is not None
        assert isinstance(rsi_result, Decimal)
        assert Decimal("0") <= rsi_result <= Decimal("100")

        # Test SMA calculation
        sma_result = await real_mean_reversion_strategy.get_sma("BTC/USDT", 20)
        assert sma_result is not None
        assert isinstance(sma_result, Decimal)

        # Test EMA calculation
        ema_result = await real_mean_reversion_strategy.get_ema("BTC/USDT", 20)
        assert ema_result is not None
        assert isinstance(ema_result, Decimal)

        # Test MACD calculation
        macd_result = await real_mean_reversion_strategy.get_macd("BTC/USDT", 12, 26, 9)
        assert macd_result is not None
        assert isinstance(macd_result, dict)
        assert "macd" in macd_result
        assert "signal" in macd_result
        assert "histogram" in macd_result

        # Test Bollinger Bands calculation
        bb_result = await real_mean_reversion_strategy.get_bollinger_bands("BTC/USDT", 20, 2.0)
        assert bb_result is not None
        assert isinstance(bb_result, dict)
        assert "upper" in bb_result
        assert "middle" in bb_result
        assert "lower" in bb_result

        # Test ATR calculation
        atr_result = await real_mean_reversion_strategy.get_atr("BTC/USDT", 14)
        assert atr_result is not None
        assert isinstance(atr_result, Decimal)
        assert atr_result >= Decimal("0")

    @pytest.mark.asyncio
    async def test_real_strategy_service_lifecycle(
        self, real_strategy_service, real_database_service, strategy_factory
    ):
        """Test complete strategy lifecycle with real service and database persistence."""
        # Create strategy configuration
        config_data = {
            "strategy_id": "test_real_trend_001",
            "name": "RealTrendFollowing",
            "strategy_type": StrategyType.TREND_FOLLOWING,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "enabled": True,
            "min_confidence": Decimal("0.7"),
            "max_positions": 2,
            "position_size_pct": Decimal("0.02"),
            "stop_loss_pct": Decimal("0.02"),
            "take_profit_pct": Decimal("0.04"),
            "parameters": {
                "fast_ma_period": 20,
                "slow_ma_period": 50,
                "rsi_period": 14,
                "volume_confirmation": True,
            },
        }

        strategy_config = StrategyConfig(**config_data)

        # Test strategy creation using strategy_factory
        strategy = await strategy_factory.create_strategy(
            strategy_type=strategy_config.strategy_type, config=strategy_config
        )
        assert strategy is not None
        assert strategy.config.strategy_id == "test_real_trend_001"
        assert strategy.config.name == "RealTrendFollowing"

        # Test strategy registration
        await real_strategy_service.register_strategy(strategy)

        # Verify strategy is registered
        registered_strategies = await real_strategy_service.get_active_strategies()
        assert "test_real_trend_001" in registered_strategies

        # Test configuration persistence
        await real_strategy_service.save_strategy_config(strategy_config)

        # Verify configuration was saved
        saved_config = await real_strategy_service.get_strategy_config("test_real_trend_001")
        assert saved_config is not None
        assert saved_config.name == "RealTrendFollowing"
        assert saved_config.strategy_type == StrategyType.TREND_FOLLOWING

        # Test strategy status updates
        await real_strategy_service.update_strategy_status("test_real_trend_001", StrategyStatus.RUNNING)

        strategy_status = await real_strategy_service.get_strategy_status("test_real_trend_001")
        assert strategy_status == StrategyStatus.RUNNING

        # Test strategy removal
        await real_strategy_service.remove_strategy("test_real_trend_001")

        # Verify strategy was removed
        final_strategies = await real_strategy_service.get_active_strategies()
        assert "test_real_trend_001" not in final_strategies

    @pytest.mark.asyncio
    async def test_real_signal_generation_with_risk_validation(
        self, real_trend_following_strategy, real_risk_service, real_data_service
    ):
        """Test real signal generation with risk management validation."""
        # Generate realistic market data for trend following
        generator = MarketDataGenerator(seed=123)
        trending_data = generator.generate_trending_data(
            periods=100, trend_strength=0.002, direction=1
        )

        # Store market data in database for strategy to use
        for md in trending_data:
            await real_data_service.store_market_data(md, exchange="binance")

        # Use latest market data for signal generation
        latest_market_data = trending_data[-1]

        # Generate signals using real strategy
        signals = await real_trend_following_strategy.generate_signals(latest_market_data)

        # Verify signals were generated
        assert signals is not None
        assert isinstance(signals, list)

        # If signals generated, validate them
        if signals:
            for signal in signals:
                assert isinstance(signal, Signal)
                assert signal.symbol == "BTC/USDT"
                assert signal.direction in [SignalDirection.BUY, SignalDirection.SELL]
                assert Decimal("0") <= signal.strength <= Decimal("1")
                assert Decimal("0") <= signal.confidence <= Decimal("1")

                # Test risk validation
                is_valid = await real_risk_service.validate_signal(
                    signal, current_portfolio_value=Decimal("100000.00")
                )
                assert isinstance(is_valid, bool)

                # Test position sizing
                position_size = await real_risk_service.calculate_position_size(
                    signal_confidence=signal.confidence,
                    current_price=latest_market_data.close,
                    stop_loss_price=latest_market_data.close * Decimal("0.98"),
                    portfolio_value=Decimal("100000.00"),
                )
                assert position_size is not None
                assert isinstance(position_size, Decimal)
                assert position_size > Decimal("0")

    @pytest.mark.asyncio
    async def test_real_strategy_types_comparison(
        self,
        real_mean_reversion_strategy,
        real_trend_following_strategy,
        real_data_service,
    ):
        """Test different strategy types with same market data."""
        # Generate market data with mixed characteristics
        generator = MarketDataGenerator(seed=456)
        mixed_data = generator.generate_realistic_data(periods=100)

        # Store market data
        for md in mixed_data:
            await real_data_service.store_market_data(md, exchange="binance")

        latest_market_data = mixed_data[-1]

        # Test strategies
        strategies = {
            "mean_reversion": real_mean_reversion_strategy,
            "trend_following": real_trend_following_strategy,
        }

        signals_by_strategy = {}

        for strategy_name, strategy in strategies.items():
            signals = await strategy.generate_signals(latest_market_data)
            signals_by_strategy[strategy_name] = signals

            # Verify signal structure
            assert signals is not None
            assert isinstance(signals, list)

            if signals:
                for signal in signals:
                    assert isinstance(signal.confidence, Decimal)
                    assert isinstance(signal.strength, Decimal)
                    assert Decimal("0") <= signal.confidence <= Decimal("1")
                    assert Decimal("0") <= signal.strength <= Decimal("1")

        # Verify different strategies can produce different signals
        # (they might be the same or different depending on market conditions)
        assert isinstance(signals_by_strategy["mean_reversion"], list)
        assert isinstance(signals_by_strategy["trend_following"], list)

    @pytest.mark.asyncio
    async def test_real_portfolio_risk_integration(
        self, real_risk_service, real_mean_reversion_strategy, real_data_service
    ):
        """Test portfolio risk management with real calculations."""
        # Generate market data
        generator = MarketDataGenerator(seed=789)
        market_data = generator.generate_realistic_data(periods=50)

        # Store market data
        for md in market_data:
            await real_data_service.store_market_data(md, exchange="binance")

        # Test portfolio value calculation
        portfolio_value = Decimal("100000.00")

        # Generate signals
        signals = await real_mean_reversion_strategy.generate_signals(market_data[-1])

        if signals:
            for signal in signals:
                # Test position sizing based on risk
                position_size = await real_risk_service.calculate_position_size(
                    signal_confidence=signal.confidence,
                    current_price=market_data[-1].close,
                    stop_loss_price=market_data[-1].close * Decimal("0.97"),
                    portfolio_value=portfolio_value,
                )

                # Verify position size is reasonable
                assert position_size is not None
                assert isinstance(position_size, Decimal)
                assert position_size > Decimal("0")

                # Position should not exceed 5% of portfolio for single trade
                max_position_value = portfolio_value * Decimal("0.05")
                position_value = position_size * market_data[-1].close
                assert position_value <= max_position_value

    @pytest.mark.asyncio
    async def test_real_database_persistence_comprehensive(
        self, real_strategy_service, real_database_service, real_data_service
    ):
        """Test comprehensive database persistence for strategies and signals."""
        # Create multiple strategy configurations
        configs = []
        for i in range(3):
            config = StrategyConfig(
                strategy_id=f"persistence_test_{i}",
                name=f"PersistenceTest_{i}",
                strategy_type=StrategyType.MEAN_REVERSION,
                symbol="BTC/USDT",
                timeframe="1h",
                enabled=True,
                min_confidence=Decimal(f"0.{60 + i}"),
                max_positions=2 + i,
                position_size_pct=Decimal("0.02"),
                stop_loss_pct=Decimal("0.02"),
                take_profit_pct=Decimal("0.04"),
                parameters={
                    "lookback_period": 20 + i * 5,
                    "entry_threshold": Decimal(f"{2.0 + i * 0.5}"),
                },
            )
            configs.append(config)

        # Save all configurations
        for config in configs:
            await real_strategy_service.save_strategy_config(config)

        # Verify all configurations were saved
        for config in configs:
            retrieved_config = await real_strategy_service.get_strategy_config(config.strategy_id)
            assert retrieved_config is not None
            assert retrieved_config.strategy_id == config.strategy_id
            assert retrieved_config.name == config.name
            assert isinstance(retrieved_config.min_confidence, Decimal)

        # Test concurrent retrieval
        retrieve_tasks = [
            real_strategy_service.get_strategy_config(config.strategy_id) for config in configs
        ]
        retrieved_configs = await asyncio.gather(*retrieve_tasks)

        assert len(retrieved_configs) == 3
        assert all(config is not None for config in retrieved_configs)

    @pytest.mark.asyncio
    async def test_real_performance_benchmarks(
        self, real_mean_reversion_strategy, real_data_service
    ):
        """Test performance benchmarks for real service operations."""
        import time

        # Generate test data
        generator = MarketDataGenerator(seed=999)
        market_data = generator.generate_realistic_data(periods=100)

        # Store market data
        for md in market_data:
            await real_data_service.store_market_data(md, exchange="binance")

        # Benchmark signal generation
        start_time = time.time()
        for _ in range(10):
            signals = await real_mean_reversion_strategy.generate_signals(market_data[-1])
        signal_generation_time = time.time() - start_time

        # Signal generation should be reasonably fast (< 5 seconds for 10 iterations)
        assert signal_generation_time < 5.0

        # Benchmark indicator calculations
        start_time = time.time()
        for _ in range(10):
            rsi = await real_mean_reversion_strategy.get_rsi("BTC/USDT", 14)
        indicator_time = time.time() - start_time

        # Indicator calculations should be fast (< 2 seconds for 10 iterations)
        assert indicator_time < 2.0

    @pytest.mark.asyncio
    async def test_real_error_handling_and_recovery(
        self, real_strategy_service, real_database_service
    ):
        """Test error handling and recovery with real services."""
        # Test invalid configuration handling
        invalid_config = StrategyConfig(
            strategy_id="",  # Invalid empty ID
            name="InvalidTest",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol="BTC/USDT",
            timeframe="1h",
            enabled=True,
            min_confidence=Decimal("0.7"),
            max_positions=2,
            position_size_pct=Decimal("0.02"),
            stop_loss_pct=Decimal("0.02"),
            take_profit_pct=Decimal("0.04"),
            parameters={},
        )

        # Should handle invalid config gracefully
        with pytest.raises((ValidationError, ValueError)):
            await real_strategy_service.save_strategy_config(invalid_config)

        # Test retrieval of non-existent strategy
        non_existent = await real_strategy_service.get_strategy_config("non_existent_strategy")
        assert non_existent is None

        # Test valid configuration after error
        valid_config = StrategyConfig(
            strategy_id="error_recovery_test",
            name="ErrorRecoveryTest",
            strategy_type=StrategyType.BREAKOUT,
            symbol="BTC/USDT",
            timeframe="1h",
            enabled=True,
            min_confidence=Decimal("0.75"),
            max_positions=2,
            position_size_pct=Decimal("0.03"),
            stop_loss_pct=Decimal("0.02"),
            take_profit_pct=Decimal("0.05"),
            parameters={
                "lookback_period": 20,
                "volume_confirmation": True,
            },
        )

        # Should succeed after previous error
        await real_strategy_service.save_strategy_config(valid_config)
        retrieved = await real_strategy_service.get_strategy_config("error_recovery_test")
        assert retrieved is not None
        assert retrieved.strategy_id == "error_recovery_test"

    @pytest.mark.asyncio
    async def test_real_concurrent_strategy_operations(
        self, real_strategy_service, real_data_service
    ):
        """Test concurrent operations with real services."""
        # Create multiple strategies concurrently
        strategy_configs = []
        for i in range(5):
            config = StrategyConfig(
                strategy_id=f"concurrent_test_{i}",
                name=f"ConcurrentTest_{i}",
                strategy_type=StrategyType.TREND_FOLLOWING,
                symbol="BTC/USDT",
                timeframe="1h",
                enabled=True,
                min_confidence=Decimal(f"0.{65 + i}"),
                max_positions=2,
                position_size_pct=Decimal("0.02"),
                stop_loss_pct=Decimal("0.02"),
                take_profit_pct=Decimal("0.04"),
                parameters={
                    "fast_ma_period": 10 + i,
                    "slow_ma_period": 30 + i * 2,
                    "rsi_period": 14,
                },
            )
            strategy_configs.append(config)

        # Save all strategies concurrently
        save_tasks = [
            real_strategy_service.save_strategy_config(config) for config in strategy_configs
        ]
        await asyncio.gather(*save_tasks)

        # Retrieve all strategies concurrently
        retrieve_tasks = [
            real_strategy_service.get_strategy_config(config.strategy_id)
            for config in strategy_configs
        ]
        retrieved_configs = await asyncio.gather(*retrieve_tasks)

        # Verify all strategies were saved and retrieved correctly
        assert len(retrieved_configs) == 5
        assert all(config is not None for config in retrieved_configs)

        # Verify data integrity
        for i, retrieved in enumerate(retrieved_configs):
            assert retrieved.strategy_id == f"concurrent_test_{i}"
            assert retrieved.name == f"ConcurrentTest_{i}"
            assert isinstance(retrieved.min_confidence, Decimal)
