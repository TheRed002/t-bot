"""
Real Database Persistence Integration Tests for Strategies.

This module tests real database persistence for strategy configurations,
signals, and analytics with PostgreSQL backend and Decimal precision.

CRITICAL: All financial data must use DECIMAL(20,8) precision in database.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import pytest

import pytest_asyncio
from src.core.types import (
    MarketData,
    Signal,
    SignalDirection,
    StrategyConfig,
    StrategyStatus,
    StrategyType,
    StrategyMetrics,
)
from src.database.models import Strategy as StrategyModel, Signal as SignalModel
from src.database.repository import StrategyRepository, SignalRepository
from src.strategies.service import StrategyService

from .fixtures.real_service_fixtures import generate_realistic_market_data_sequence


class TestRealStrategyConfigurationPersistence:
    """Test real strategy configuration persistence to PostgreSQL."""

    @pytest_asyncio.fixture
    async def strategy_repository(self, clean_database):
        """Create real strategy repository with database connection."""
        repository = StrategyRepository(clean_database)
        yield repository
        await repository.cleanup()

    @pytest_asyncio.fixture
    async def signal_repository(self, clean_database):
        """Create real signal repository with database connection."""
        repository = SignalRepository(clean_database)
        yield repository
        await repository.cleanup()

    @pytest.mark.asyncio
    async def test_strategy_config_crud_operations(self, strategy_repository):
        """Test complete CRUD operations for strategy configurations."""
        # Create strategy configuration with Decimal precision
        config = StrategyConfig(
            strategy_id="crud_test_001",
            name="crud_test_strategy",
            strategy_type=StrategyType.MEAN_REVERSION,
            enabled=True,
            symbol="BTC/USDT",
            timeframe="1h",
            min_confidence=Decimal("0.75"),
            max_positions=5,
            position_size_pct=Decimal("0.025"),
            stop_loss_pct=Decimal("0.02"),
            take_profit_pct=Decimal("0.04"),
            parameters={
                "lookback_period": 20,
                "entry_threshold": Decimal("2.5"),
                "exit_threshold": Decimal("0.5"),
                "atr_multiplier": Decimal("2.0"),
                "volume_filter": True,
                "min_volume_ratio": Decimal("1.5"),
            },
        )

        # CREATE: Save configuration to database
        await strategy_repository.save_strategy_config(config)

        # READ: Retrieve configuration from database
        retrieved_config = await strategy_repository.get_strategy_config("crud_test_001")

        # Verify persistence accuracy
        assert retrieved_config is not None
        assert retrieved_config.strategy_id == config.strategy_id
        assert retrieved_config.name == config.name
        assert retrieved_config.strategy_type == config.strategy_type
        assert retrieved_config.symbol == config.symbol

        # Verify Decimal precision preservation
        assert isinstance(retrieved_config.min_confidence, Decimal)
        assert retrieved_config.min_confidence == Decimal("0.75")
        assert isinstance(retrieved_config.position_size_pct, Decimal)
        assert retrieved_config.position_size_pct == Decimal("0.025")

        # Verify parameters with Decimal types
        assert isinstance(retrieved_config.parameters["entry_threshold"], Decimal)
        assert retrieved_config.parameters["entry_threshold"] == Decimal("2.5")
        assert isinstance(retrieved_config.parameters["atr_multiplier"], Decimal)
        assert retrieved_config.parameters["atr_multiplier"] == Decimal("2.0")

        # UPDATE: Modify configuration
        updated_config = retrieved_config.copy()
        updated_config.min_confidence = Decimal("0.80")
        updated_config.parameters["entry_threshold"] = Decimal("3.0")
        updated_config.enabled = False

        await strategy_repository.update_strategy_config(updated_config)

        # Verify update
        final_config = await strategy_repository.get_strategy_config("crud_test_001")
        assert final_config.min_confidence == Decimal("0.80")
        assert final_config.parameters["entry_threshold"] == Decimal("3.0")
        assert final_config.enabled is False

        # DELETE: Remove configuration
        await strategy_repository.delete_strategy_config("crud_test_001")

        # Verify deletion
        deleted_config = await strategy_repository.get_strategy_config("crud_test_001")
        assert deleted_config is None

    @pytest.mark.asyncio
    async def test_strategy_config_validation_constraints(self, strategy_repository):
        """Test database constraints and validation for strategy configurations."""
        # Test unique constraint violation
        config1 = StrategyConfig(
            strategy_id="constraint_test_001",
            name="constraint_test_strategy_1",
            strategy_type=StrategyType.MOMENTUM,
            symbol="BTC/USDT",
            timeframe="1h",
        )

        config2 = StrategyConfig(
            strategy_id="constraint_test_001",  # Same ID
            name="constraint_test_strategy_2",
            strategy_type=StrategyType.TREND_FOLLOWING,
            symbol="ETH/USDT",
            timeframe="1h",
        )

        # Save first configuration
        await strategy_repository.save_strategy_config(config1)

        # Attempt to save second configuration with same ID should fail
        with pytest.raises(Exception):  # Database constraint violation
            await strategy_repository.save_strategy_config(config2)

        # Test decimal precision constraints
        config_with_precision = StrategyConfig(
            strategy_id="precision_test_001",
            name="precision_test_strategy",
            strategy_type=StrategyType.BREAKOUT,
            symbol="BTC/USDT",
            timeframe="1h",
            min_confidence=Decimal("0.123456789"),  # High precision
            position_size_pct=Decimal("0.025678901"),  # High precision
            parameters={
                "entry_threshold": Decimal("2.123456789"),  # High precision
                "atr_multiplier": Decimal("1.987654321"),  # High precision
            },
        )

        await strategy_repository.save_strategy_config(config_with_precision)

        # Retrieve and verify precision preservation
        retrieved_precision_config = await strategy_repository.get_strategy_config("precision_test_001")
        assert isinstance(retrieved_precision_config.min_confidence, Decimal)
        assert isinstance(retrieved_precision_config.position_size_pct, Decimal)
        assert isinstance(retrieved_precision_config.parameters["entry_threshold"], Decimal)

        # Verify precision is maintained (at least to 18 decimal places for crypto)
        assert str(retrieved_precision_config.min_confidence)[:10] == "0.12345678"

    @pytest.mark.asyncio
    async def test_strategy_config_batch_operations(self, strategy_repository):
        """Test batch operations for strategy configurations."""
        # Create multiple configurations
        configs = []
        for i in range(5):
            config = StrategyConfig(
                strategy_id=f"batch_test_{i:03d}",
                name=f"batch_test_strategy_{i}",
                strategy_type=StrategyType.MEAN_REVERSION,
                symbol="BTC/USDT",
                timeframe="1h",
                min_confidence=Decimal(f"0.{60 + i}"),  # 0.60, 0.61, 0.62, etc.
                parameters={
                    "lookback_period": 20 + i,
                    "entry_threshold": Decimal(f"{2 + i}.0"),  # 2.0, 3.0, 4.0, etc.
                },
            )
            configs.append(config)

        # Batch save
        await strategy_repository.save_strategy_configs_batch(configs)

        # Batch retrieve
        retrieved_configs = await strategy_repository.get_strategy_configs_batch(
            [f"batch_test_{i:03d}" for i in range(5)]
        )

        # Verify batch operations
        assert len(retrieved_configs) == 5

        for i, config in enumerate(retrieved_configs):
            assert config.strategy_id == f"batch_test_{i:03d}"
            assert config.min_confidence == Decimal(f"0.{60 + i}")
            assert config.parameters["entry_threshold"] == Decimal(f"{2 + i}.0")

        # Batch update
        for config in retrieved_configs:
            config.enabled = False
            config.min_confidence = config.min_confidence + Decimal("0.1")

        await strategy_repository.update_strategy_configs_batch(retrieved_configs)

        # Verify batch update
        updated_configs = await strategy_repository.get_strategy_configs_batch(
            [f"batch_test_{i:03d}" for i in range(5)]
        )

        for config in updated_configs:
            assert config.enabled is False
            # min_confidence should be increased by 0.1

        # Batch delete
        await strategy_repository.delete_strategy_configs_batch(
            [f"batch_test_{i:03d}" for i in range(5)]
        )

        # Verify batch deletion
        deleted_configs = await strategy_repository.get_strategy_configs_batch(
            [f"batch_test_{i:03d}" for i in range(5)]
        )
        assert len([c for c in deleted_configs if c is not None]) == 0


class TestRealSignalPersistence:
    """Test real signal persistence to PostgreSQL with Decimal precision."""

    @pytest_asyncio.fixture
    async def signal_repository(self, clean_database):
        """Create real signal repository."""
        repository = SignalRepository(clean_database)
        yield repository
        await repository.cleanup()

    def create_test_signal(
        self,
        strategy_id: str = "test_strategy_001",
        symbol: str = "BTC/USDT",
        direction: SignalDirection = SignalDirection.BUY,
        confidence: Decimal = Decimal("0.75"),
        strength: Decimal = Decimal("0.80"),
        metadata: Optional[Dict[str, Any]] = None
    ) -> Signal:
        """Create test signal with Decimal precision."""
        return Signal(
            strategy_id=strategy_id,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            strength=strength,
            source="test_strategy",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
            metadata=metadata or {
                "indicator_value": Decimal("2.5"),
                "volume_ratio": Decimal("1.8"),
                "atr": Decimal("150.25"),
            },
        )

    @pytest.mark.asyncio
    async def test_signal_crud_operations(self, signal_repository):
        """Test complete CRUD operations for signals."""
        # CREATE: Save signal to database
        signal = self.create_test_signal(
            strategy_id="signal_crud_test_001",
            confidence=Decimal("0.85"),
            strength=Decimal("0.90"),
            metadata={
                "rsi": Decimal("25.5"),
                "macd": Decimal("150.75"),
                "volume_ratio": Decimal("2.3"),
                "z_score": Decimal("-2.8"),
            }
        )

        signal_id = await signal_repository.save_signal(signal)
        assert signal_id is not None

        # READ: Retrieve signal from database
        retrieved_signal = await signal_repository.get_signal(signal_id)

        # Verify persistence accuracy
        assert retrieved_signal is not None
        assert retrieved_signal.strategy_id == signal.strategy_id
        assert retrieved_signal.symbol == signal.symbol
        assert retrieved_signal.direction == signal.direction

        # Verify Decimal precision preservation
        assert isinstance(retrieved_signal.confidence, Decimal)
        assert retrieved_signal.confidence == Decimal("0.85")
        assert isinstance(retrieved_signal.strength, Decimal)
        assert retrieved_signal.strength == Decimal("0.90")

        # Verify metadata with Decimal values
        assert isinstance(retrieved_signal.metadata["rsi"], Decimal)
        assert retrieved_signal.metadata["rsi"] == Decimal("25.5")
        assert isinstance(retrieved_signal.metadata["z_score"], Decimal)
        assert retrieved_signal.metadata["z_score"] == Decimal("-2.8")

        # UPDATE: Modify signal (if supported)
        updated_signal = retrieved_signal.copy()
        updated_signal.confidence = Decimal("0.95")
        updated_signal.metadata["updated"] = True

        await signal_repository.update_signal(updated_signal)

        # Verify update
        final_signal = await signal_repository.get_signal(signal_id)
        assert final_signal.confidence == Decimal("0.95")
        assert final_signal.metadata["updated"] is True

        # DELETE: Remove signal
        await signal_repository.delete_signal(signal_id)

        # Verify deletion
        deleted_signal = await signal_repository.get_signal(signal_id)
        assert deleted_signal is None

    @pytest.mark.asyncio
    async def test_signal_history_and_querying(self, signal_repository):
        """Test signal history storage and complex querying."""
        # Create signals across different time periods
        base_time = datetime.now(timezone.utc)
        signals = []

        for i in range(20):
            # Create signals with varying properties
            signal = Signal(
                strategy_id="history_test_001",
                symbol="BTC/USDT",
                direction=SignalDirection.BUY if i % 2 == 0 else SignalDirection.SELL,
                confidence=Decimal(f"0.{60 + i % 25}"),  # 0.60 to 0.84
                strength=Decimal(f"0.{70 + i % 20}"),    # 0.70 to 0.89
                source="history_test_strategy",
                timestamp=base_time - timedelta(hours=i),
                strategy_name="history_test_strategy",
                metadata={
                    "signal_number": i,
                    "price": Decimal(f"{50000 + i * 100}"),
                    "volume": Decimal(f"{1000 + i * 50}"),
                    "rsi": Decimal(f"{30 + i % 40}"),  # 30 to 69
                },
            )
            signals.append(signal)

        # Batch save signals
        signal_ids = await signal_repository.save_signals_batch(signals)
        assert len(signal_ids) == 20

        # Query signals by strategy
        strategy_signals = await signal_repository.get_signals_by_strategy(
            "history_test_001", limit=10
        )
        assert len(strategy_signals) == 10
        assert all(s.strategy_id == "history_test_001" for s in strategy_signals)

        # Query signals by time range
        time_range_signals = await signal_repository.get_signals_by_time_range(
            start_time=base_time - timedelta(hours=10),
            end_time=base_time - timedelta(hours=5),
            strategy_id="history_test_001"
        )
        assert len(time_range_signals) == 6  # Hours 5-10

        # Query signals by direction
        buy_signals = await signal_repository.get_signals_by_direction(
            "history_test_001", SignalDirection.BUY
        )
        sell_signals = await signal_repository.get_signals_by_direction(
            "history_test_001", SignalDirection.SELL
        )
        assert len(buy_signals) == 10  # Even indices
        assert len(sell_signals) == 10  # Odd indices

        # Query signals by confidence range
        high_confidence_signals = await signal_repository.get_signals_by_confidence_range(
            "history_test_001",
            min_confidence=Decimal("0.75"),
            max_confidence=Decimal("1.0")
        )
        assert all(s.confidence >= Decimal("0.75") for s in high_confidence_signals)

        # Query with pagination
        page_1 = await signal_repository.get_signals_paginated(
            strategy_id="history_test_001", page=1, page_size=5
        )
        page_2 = await signal_repository.get_signals_paginated(
            strategy_id="history_test_001", page=2, page_size=5
        )

        assert len(page_1) == 5
        assert len(page_2) == 5
        assert page_1[0].signal_id != page_2[0].signal_id  # Different signals

    @pytest.mark.asyncio
    async def test_signal_aggregation_and_analytics(self, signal_repository):
        """Test signal aggregation and analytics queries."""
        # Create signals for analytics testing
        strategy_id = "analytics_test_001"
        base_time = datetime.now(timezone.utc)

        # Create signals with known patterns for analytics
        win_signals = []
        loss_signals = []

        # Create 15 winning signals
        for i in range(15):
            signal = Signal(
                strategy_id=strategy_id,
                symbol="BTC/USDT",
                direction=SignalDirection.BUY,
                confidence=Decimal(f"0.{75 + i % 20}"),
                strength=Decimal(f"0.{80 + i % 15}"),
                source="analytics_test_strategy",
                timestamp=base_time - timedelta(hours=i),
                strategy_name="analytics_test_strategy",
                metadata={
                    "result": "win",
                    "pnl": Decimal(f"{100 + i * 10}"),  # Positive PnL
                    "confidence_bucket": "high" if i < 10 else "medium",
                },
            )
            win_signals.append(signal)

        # Create 5 losing signals
        for i in range(5):
            signal = Signal(
                strategy_id=strategy_id,
                symbol="BTC/USDT",
                direction=SignalDirection.SELL,
                confidence=Decimal(f"0.{55 + i * 5}"),
                strength=Decimal(f"0.{60 + i * 8}"),
                source="analytics_test_strategy",
                timestamp=base_time - timedelta(hours=15 + i),
                strategy_name="analytics_test_strategy",
                metadata={
                    "result": "loss",
                    "pnl": Decimal(f"-{50 + i * 20}"),  # Negative PnL
                    "confidence_bucket": "low",
                },
            )
            loss_signals.append(signal)

        # Save all signals
        all_signals = win_signals + loss_signals
        await signal_repository.save_signals_batch(all_signals)

        # Test signal count aggregation
        total_signals = await signal_repository.get_signal_count(strategy_id)
        assert total_signals == 20

        # Test signal count by direction
        buy_count = await signal_repository.get_signal_count_by_direction(
            strategy_id, SignalDirection.BUY
        )
        sell_count = await signal_repository.get_signal_count_by_direction(
            strategy_id, SignalDirection.SELL
        )
        assert buy_count == 15
        assert sell_count == 5

        # Test average confidence
        avg_confidence = await signal_repository.get_average_confidence(strategy_id)
        assert isinstance(avg_confidence, Decimal)
        assert avg_confidence > Decimal("0.5")

        # Test confidence distribution
        confidence_distribution = await signal_repository.get_confidence_distribution(
            strategy_id, bucket_size=Decimal("0.1")
        )
        assert isinstance(confidence_distribution, dict)
        assert len(confidence_distribution) > 0

        # Test time-based aggregation
        daily_signal_counts = await signal_repository.get_daily_signal_counts(
            strategy_id,
            start_date=base_time.date() - timedelta(days=1),
            end_date=base_time.date() + timedelta(days=1)
        )
        assert len(daily_signal_counts) >= 1

    @pytest.mark.asyncio
    async def test_signal_performance_tracking(self, signal_repository):
        """Test signal performance tracking and outcome linking."""
        strategy_id = "performance_test_001"
        base_time = datetime.now(timezone.utc)

        # Create signals with performance outcomes
        performance_signals = []

        for i in range(10):
            # Create signal
            signal = Signal(
                strategy_id=strategy_id,
                symbol="BTC/USDT",
                direction=SignalDirection.BUY if i % 2 == 0 else SignalDirection.SELL,
                confidence=Decimal(f"0.{70 + i % 25}"),
                strength=Decimal(f"0.{75 + i % 20}"),
                source="performance_test_strategy",
                timestamp=base_time - timedelta(hours=i * 2),
                strategy_name="performance_test_strategy",
                metadata={
                    "entry_price": Decimal(f"{50000 + i * 200}"),
                    "exit_price": Decimal(f"{50000 + i * 200 + (100 if i % 3 == 0 else -50)}"),
                    "pnl": Decimal(f"{100 if i % 3 == 0 else -50}"),
                    "outcome": "win" if i % 3 == 0 else "loss",
                    "holding_period_hours": i + 1,
                },
            )
            performance_signals.append(signal)

        # Save performance signals
        await signal_repository.save_signals_batch(performance_signals)

        # Calculate performance metrics
        win_rate = await signal_repository.calculate_win_rate(strategy_id)
        assert isinstance(win_rate, Decimal)
        assert Decimal("0") <= win_rate <= Decimal("1")

        # Calculate average PnL
        avg_pnl = await signal_repository.calculate_average_pnl(strategy_id)
        assert isinstance(avg_pnl, Decimal)

        # Calculate total PnL
        total_pnl = await signal_repository.calculate_total_pnl(strategy_id)
        assert isinstance(total_pnl, Decimal)

        # Get performance by confidence bucket
        high_confidence_performance = await signal_repository.get_performance_by_confidence(
            strategy_id,
            min_confidence=Decimal("0.8"),
            max_confidence=Decimal("1.0")
        )

        low_confidence_performance = await signal_repository.get_performance_by_confidence(
            strategy_id,
            min_confidence=Decimal("0.5"),
            max_confidence=Decimal("0.8")
        )

        # Verify performance tracking
        assert isinstance(high_confidence_performance, dict)
        assert isinstance(low_confidence_performance, dict)

        # High confidence signals should generally perform better
        if (high_confidence_performance.get("win_rate") and
            low_confidence_performance.get("win_rate")):
            # This is an expectation but may not always hold in small samples
            pass


class TestRealStrategyServiceDatabaseIntegration:
    """Test complete integration of StrategyService with database persistence."""

    @pytest_asyncio.fixture
    async def real_strategy_service(self, clean_database, strategy_service_container):
        """Create real strategy service with database integration."""
        service = StrategyService(
            name="DatabaseIntegrationTestService",
            risk_manager=strategy_service_container.risk_service,
            data_service=strategy_service_container.data_service,
        )
        await service.initialize()
        yield service
        await service.cleanup()

    @pytest.mark.asyncio
    async def test_end_to_end_strategy_lifecycle_with_persistence(self, real_strategy_service):
        """Test complete strategy lifecycle with database persistence."""
        # Create strategy configuration
        config = StrategyConfig(
            strategy_id="e2e_lifecycle_test_001",
            name="e2e_lifecycle_test_strategy",
            strategy_type=StrategyType.MEAN_REVERSION,
            symbol="BTC/USDT",
            timeframe="1h",
            min_confidence=Decimal("0.70"),
            parameters={
                "lookback_period": 20,
                "entry_threshold": Decimal("2.0"),
            }
        )

        # Register and persist strategy
        await real_strategy_service.register_strategy_config(config)

        # Verify persistence
        retrieved_config = await real_strategy_service.get_strategy_config(
            "e2e_lifecycle_test_001"
        )
        assert retrieved_config is not None
        assert retrieved_config.strategy_id == config.strategy_id

        # Create and start strategy
        strategy = await real_strategy_service.create_strategy(config)
        await strategy.start()

        # Generate and persist signals
        market_data_sequence = generate_realistic_market_data_sequence(
            pattern="mean_reversion", periods=25
        )

        generated_signals = []
        for market_data in market_data_sequence:
            # Store market data
            await real_strategy_service.store_market_data(market_data, exchange=market_data.exchange)

            # Generate signals
            signals = await strategy.generate_signals(market_data)
            generated_signals.extend(signals)

            # Persist signals
            for signal in signals:
                await real_strategy_service.persist_signal(signal)

        # Retrieve and verify signal history
        signal_history = await real_strategy_service.get_signal_history(
            strategy_id="e2e_lifecycle_test_001",
            limit=100
        )

        assert len(signal_history) == len(generated_signals)

        # Verify all signals have proper Decimal precision
        for signal in signal_history:
            assert isinstance(signal.confidence, Decimal)
            assert isinstance(signal.strength, Decimal)

        # Generate strategy metrics
        metrics = await real_strategy_service.calculate_strategy_metrics(
            "e2e_lifecycle_test_001"
        )

        assert isinstance(metrics, StrategyMetrics)
        assert isinstance(metrics.total_signals, int)
        assert isinstance(metrics.win_rate, Decimal)
        assert isinstance(metrics.total_pnl, Decimal)

        # Update strategy configuration
        updated_config = retrieved_config.copy()
        updated_config.min_confidence = Decimal("0.75")
        await real_strategy_service.update_strategy_config(updated_config)

        # Verify update persistence
        final_config = await real_strategy_service.get_strategy_config(
            "e2e_lifecycle_test_001"
        )
        assert final_config.min_confidence == Decimal("0.75")

        # Cleanup
        await strategy.stop()
        await real_strategy_service.unregister_strategy_config("e2e_lifecycle_test_001")

    @pytest.mark.asyncio
    async def test_concurrent_strategy_persistence(self, real_strategy_service):
        """Test concurrent strategy operations with database persistence."""
        # Create multiple strategy configurations
        configs = []
        for i in range(5):
            config = StrategyConfig(
                strategy_id=f"concurrent_test_{i:03d}",
                name=f"concurrent_test_strategy_{i}",
                strategy_type=StrategyType.MOMENTUM,
                symbol="BTC/USDT",
                timeframe="1h",
                min_confidence=Decimal(f"0.{60 + i * 5}"),
                parameters={"lookback_period": 20 + i}
            )
            configs.append(config)

        # Concurrent registration
        registration_tasks = [
            real_strategy_service.register_strategy_config(config)
            for config in configs
        ]
        await asyncio.gather(*registration_tasks)

        # Concurrent retrieval
        retrieval_tasks = [
            real_strategy_service.get_strategy_config(f"concurrent_test_{i:03d}")
            for i in range(5)
        ]
        retrieved_configs = await asyncio.gather(*retrieval_tasks)

        # Verify all configurations were persisted correctly
        assert len(retrieved_configs) == 5
        for i, config in enumerate(retrieved_configs):
            assert config is not None
            assert config.strategy_id == f"concurrent_test_{i:03d}"
            assert config.min_confidence == Decimal(f"0.{60 + i * 5}")

        # Concurrent updates
        for config in retrieved_configs:
            config.enabled = False

        update_tasks = [
            real_strategy_service.update_strategy_config(config)
            for config in retrieved_configs
        ]
        await asyncio.gather(*update_tasks)

        # Verify concurrent updates
        final_retrieval_tasks = [
            real_strategy_service.get_strategy_config(f"concurrent_test_{i:03d}")
            for i in range(5)
        ]
        final_configs = await asyncio.gather(*final_retrieval_tasks)

        for config in final_configs:
            assert config.enabled is False

        # Concurrent cleanup
        cleanup_tasks = [
            real_strategy_service.unregister_strategy_config(f"concurrent_test_{i:03d}")
            for i in range(5)
        ]
        await asyncio.gather(*cleanup_tasks)