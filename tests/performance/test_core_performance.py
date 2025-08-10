"""
Performance tests for core functionality.

These tests measure performance of core components like configuration,
logging, and type validation.
"""

import pytest
import time
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import statistics

from src.core.config import Config
from src.core.logging import setup_logging, get_logger, log_performance, log_async_performance
from src.core.types import (
    TradingMode, SignalDirection, OrderSide, OrderType,
    Signal, MarketData, OrderRequest, OrderResponse, Position
)
from src.core.exceptions import TradingBotError, ValidationError


@pytest.fixture(scope="session")
def config():
    """Provide test configuration."""
    return Config()


@pytest.fixture(scope="session")
def setup_logging_for_tests():
    """Setup logging for tests."""
    setup_logging(environment="test")


class TestConfigurationPerformance:
    """Test configuration loading and validation performance."""

    def test_config_loading_performance(self, config):
        """Test configuration loading performance."""
        start_time = time.time()

        # Load configuration multiple times
        for _ in range(100):
            test_config = Config()

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 100

        print(f"Configuration loading: {avg_time:.6f}s per load")
        # Should be under 15ms per load (increased due to complex config)
        assert avg_time < 0.015

    def test_config_validation_performance(self, config):
        """Test configuration validation performance."""
        start_time = time.time()

        # Perform validation multiple times
        for _ in range(1000):
            config.environment = "development"
            config.validate_environment("development")

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000

        print(f"Configuration validation: {avg_time:.6f}s per validation")
        assert avg_time < 0.001  # Should be under 1ms per validation

    def test_database_url_generation_performance(self, config):
        """Test database URL generation performance."""
        start_time = time.time()

        # Generate URLs multiple times
        for _ in range(1000):
            db_url = config.get_database_url()
            async_db_url = config.get_async_database_url()
            redis_url = config.get_redis_url()

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 3000  # 3 URL generations per iteration

        print(f"URL generation: {avg_time:.6f}s per URL")
        assert avg_time < 0.0001  # Should be under 0.1ms per URL


class TestLoggingPerformance:
    """Test logging performance."""

    def test_logger_creation_performance(self, setup_logging_for_tests):
        """Test logger creation performance."""
        start_time = time.time()

        # Create loggers multiple times
        for i in range(1000):
            logger = get_logger(f"test_logger_{i}")

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000

        print(f"Logger creation: {avg_time:.6f}s per logger")
        assert avg_time < 0.001  # Should be under 1ms per logger

    def test_logging_performance(self, setup_logging_for_tests):
        """Test logging performance."""
        logger = get_logger("test_performance")

        start_time = time.time()

        # Log messages multiple times
        for i in range(1000):
            logger.info(
                f"Test message {i}",
                user_id="test_user",
                operation="test")

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000

        print(f"Logging: {avg_time:.6f}s per log message")
        assert avg_time < 0.001  # Should be under 1ms per log message

    def test_performance_decorator_overhead(self, setup_logging_for_tests):
        """Test performance decorator overhead."""
        @log_performance
        def test_function():
            return "test_result"

        # Measure overhead of performance decorator
        start_time = time.time()

        for _ in range(1000):
            result = test_function()

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000

        print(f"Performance decorator overhead: {avg_time:.6f}s per call")
        assert avg_time < 0.001  # Should be under 1ms per call

    @pytest.mark.asyncio
    async def test_async_performance_decorator_overhead(
            self, setup_logging_for_tests):
        """Test async performance decorator overhead."""
        @log_async_performance
        async def test_async_function():
            return "test_async_result"

        # Measure overhead of async performance decorator
        start_time = time.time()

        for _ in range(1000):
            result = await test_async_function()

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000

        print(
            f"Async performance decorator overhead: {
                avg_time: .6f}s per call")
        assert avg_time < 0.001  # Should be under 1ms per call


class TestTypeValidationPerformance:
    """Test type validation performance."""

    def test_signal_creation_performance(self):
        """Test Signal model creation performance."""
        start_time = time.time()

        # Create signals multiple times
        for i in range(1000):
            signal = Signal(
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USDT",
                strategy_name="test_strategy",
                metadata={"test": "data"}
            )

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000

        print(f"Signal creation: {avg_time:.6f}s per signal")
        assert avg_time < 0.001  # Should be under 1ms per signal

    def test_market_data_creation_performance(self):
        """Test MarketData model creation performance."""
        start_time = time.time()

        # Create market data multiple times
        for i in range(1000):
            market_data = MarketData(
                symbol="BTC/USDT",
                price=Decimal("50000.00"),
                volume=Decimal("100.5"),
                timestamp=datetime.now(timezone.utc),
                bid=Decimal("49999.00"),
                ask=Decimal("50001.00"),
                open_price=Decimal("49900.00"),
                high_price=Decimal("50100.00"),
                low_price=Decimal("49800.00")
            )

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000

        print(f"MarketData creation: {avg_time:.6f}s per market data")
        assert avg_time < 0.001  # Should be under 1ms per market data

    def test_order_request_creation_performance(self):
        """Test OrderRequest model creation performance."""
        start_time = time.time()

        # Create order requests multiple times
        for i in range(1000):
            order_request = OrderRequest(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                time_in_force="GTC",
                client_order_id=f"test_order_{i}"
            )

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000

        print(f"OrderRequest creation: {avg_time:.6f}s per order request")
        assert avg_time < 0.001  # Should be under 1ms per order request

    def test_position_creation_performance(self):
        """Test Position model creation performance."""
        start_time = time.time()

        # Create positions multiple times
        for i in range(1000):
            position = Position(
                symbol="BTC/USDT",
                quantity=Decimal("2.0"),
                entry_price=Decimal("50000.00"),
                current_price=Decimal("51000.00"),
                unrealized_pnl=Decimal("2000.00"),
                side=OrderSide.BUY,
                timestamp=datetime.now(timezone.utc)
            )

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000

        print(f"Position creation: {avg_time:.6f}s per position")
        assert avg_time < 0.001  # Should be under 1ms per position


class TestExceptionPerformance:
    """Test exception handling performance."""

    def test_exception_creation_performance(self):
        """Test exception creation performance."""
        start_time = time.time()

        # Create exceptions multiple times
        for i in range(1000):
            error = TradingBotError(
                f"Test error message {i}",
                error_code="TEST_ERROR",
                details={"test": "data"}
            )

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000

        print(f"Exception creation: {avg_time:.6f}s per exception")
        assert avg_time < 0.001  # Should be under 1ms per exception

    def test_exception_handling_performance(self):
        """Test exception handling performance."""
        start_time = time.time()

        # Handle exceptions multiple times
        for i in range(1000):
            try:
                raise ValidationError(f"Test validation error {i}")
            except ValidationError:
                pass

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000

        print(f"Exception handling: {avg_time:.6f}s per exception")
        assert avg_time < 0.001  # Should be under 1ms per exception


class TestBulkOperationsPerformance:
    """Test bulk operations performance."""

    def test_bulk_signal_creation(self):
        """Test bulk signal creation performance."""
        signals = []
        start_time = time.time()

        # Create signals in bulk
        for i in range(10000):
            signal = Signal(
                direction=SignalDirection.BUY if i % 2 == 0 else SignalDirection.SELL,
                confidence=0.5 + (i % 50) / 100.0,
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USDT",
                strategy_name="test_strategy",
                metadata={"test": "data", "index": i}
            )
            signals.append(signal)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10000

        print(
            f"Bulk signal creation: {
                avg_time: .6f}s per signal({
                    len(signals)} signals)")
        assert avg_time < 0.001  # Should be under 1ms per signal
        assert len(signals) == 10000

    def test_bulk_market_data_creation(self):
        """Test bulk market data creation performance."""
        market_data_list = []
        start_time = time.time()

        # Create market data in bulk
        for i in range(10000):
            market_data = MarketData(
                symbol="BTC/USDT",
                price=Decimal("50000.00") + Decimal(str(i)),
                volume=Decimal("100.5") + Decimal(str(i % 100)),
                timestamp=datetime.now(timezone.utc),
                bid=Decimal("49999.00") + Decimal(str(i)),
                ask=Decimal("50001.00") + Decimal(str(i)),
                open_price=Decimal("49900.00") + Decimal(str(i)),
                high_price=Decimal("50100.00") + Decimal(str(i)),
                low_price=Decimal("49800.00") + Decimal(str(i))
            )
            market_data_list.append(market_data)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10000

        print(
            f"Bulk market data creation: {
                avg_time: .6f}s per market data({
                    len(market_data_list)} items)")
        assert avg_time < 0.001  # Should be under 1ms per market data
        assert len(market_data_list) == 10000


class TestMemoryUsagePerformance:
    """Test memory usage performance."""

    def test_memory_efficiency_small_objects(self):
        """Test memory efficiency with small objects."""
        import sys

        # Measure memory usage for small objects
        objects = []
        initial_memory = sys.getsizeof(objects)

        for i in range(10000):
            signal = Signal(
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USDT",
                strategy_name="test_strategy",
                metadata={"test": "data"}
            )
            objects.append(signal)

        final_memory = sys.getsizeof(objects)
        memory_per_object = (final_memory - initial_memory) / len(objects)

        print(f"Memory usage per signal: {memory_per_object:.2f} bytes")
        assert memory_per_object < 1000  # Should be reasonable memory usage

    def test_memory_efficiency_large_objects(self):
        """Test memory efficiency with large objects."""
        import sys

        # Measure memory usage for large objects
        objects = []
        initial_memory = sys.getsizeof(objects)

        for i in range(1000):
            market_data = MarketData(
                symbol="BTC/USDT",
                price=Decimal("50000.00"),
                volume=Decimal("100.5"),
                timestamp=datetime.now(timezone.utc),
                bid=Decimal("49999.00"),
                ask=Decimal("50001.00"),
                open_price=Decimal("49900.00"),
                high_price=Decimal("50100.00"),
                low_price=Decimal("49800.00")
            )
            objects.append(market_data)

        final_memory = sys.getsizeof(objects)
        memory_per_object = (final_memory - initial_memory) / len(objects)

        print(f"Memory usage per market data: {memory_per_object:.2f} bytes")
        assert memory_per_object < 2000  # Should be reasonable memory usage
