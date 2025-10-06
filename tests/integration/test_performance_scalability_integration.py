"""
Performance and Scalability Integration Tests

This test suite validates system performance and scalability characteristics
across integrated modules under realistic load conditions.
"""

import asyncio
import logging
import statistics
import time
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

import psutil
import pytest
import pytest_asyncio

from src.core.types import (
    MarketData,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Signal,
    SignalDirection,
)
from tests.integration.base_integration import (
    BaseIntegrationTest,
    MockExchangeFactory,
    MockStrategyFactory,
    PerformanceMonitor,
)

logger = logging.getLogger(__name__)


class PerformanceScalabilityTest(BaseIntegrationTest):
    """Performance and scalability integration tests."""

    def __init__(self):
        super().__init__()
        self.load_test_monitor = PerformanceMonitor()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_concurrent_exchange_operations(self):
        """Test concurrent operations across multiple exchanges."""
        logger.info("Testing concurrent exchange operations performance")

        # Create multiple mock exchanges
        exchanges = {
            f"exchange_{i}": MockExchangeFactory.create_binance_mock(
                initial_balance={"USDT": Decimal("100000.0"), "BTC": Decimal("0.0")},
                order_latency=0.05 + (i * 0.01),  # Varying latencies
            )
            for i in range(5)
        }

        self.load_test_monitor.start()

        async def concurrent_market_data_fetch(
            exchange_name: str, exchange: Mock, iterations: int = 20
        ):
            """Fetch market data concurrently from an exchange."""
            latencies = []
            errors = 0

            for i in range(iterations):
                start_time = time.time()
                try:
                    market_data = await exchange.get_market_data("BTC/USDT")
                    latency = time.time() - start_time
                    latencies.append(latency)

                    assert market_data is not None
                    assert market_data.symbol == "BTC/USDT"

                except Exception as e:
                    errors += 1
                    logger.error(f"Market data fetch error on {exchange_name}: {e}")

            return {
                "exchange": exchange_name,
                "iterations": iterations,
                "avg_latency": statistics.mean(latencies) if latencies else 0,
                "max_latency": max(latencies) if latencies else 0,
                "min_latency": min(latencies) if latencies else 0,
                "errors": errors,
            }

        # Run concurrent operations
        tasks = [
            concurrent_market_data_fetch(name, exchange) for name, exchange in exchanges.items()
        ]

        results = await asyncio.gather(*tasks)
        self.load_test_monitor.stop()

        # Analyze results
        total_operations = sum(r["iterations"] for r in results)
        avg_latencies = [r["avg_latency"] for r in results]
        total_errors = sum(r["errors"] for r in results)

        # Performance assertions
        assert total_operations == 100  # 5 exchanges * 20 iterations
        assert statistics.mean(avg_latencies) < 0.5  # Average latency under 500ms
        assert total_errors < total_operations * 0.05  # Error rate under 5%

        # Log performance metrics
        logger.info(f"Concurrent operations completed: {total_operations}")
        logger.info(f"Average latency: {statistics.mean(avg_latencies):.3f}s")
        logger.info(f"Error rate: {total_errors / total_operations:.1%}")

        logger.info("✅ Concurrent exchange operations test passed")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_high_frequency_order_processing(self):
        """Test high-frequency order processing performance."""
        logger.info("Testing high-frequency order processing")

        exchange = MockExchangeFactory.create_binance_mock()
        order_count = 100
        batch_size = 10

        self.load_test_monitor.start()

        async def process_order_batch(batch_id: int, orders_in_batch: int):
            """Process a batch of orders."""
            batch_results = []

            for i in range(orders_in_batch):
                order_data = {
                    "symbol": "BTC/USDT",
                    "side": "BUY" if i % 2 == 0 else "SELL",
                    "type": "MARKET",
                    "quantity": Decimal("0.01"),
                }

                start_time = time.time()
                try:
                    order_id = await exchange.place_order(order_data)
                    latency = time.time() - start_time

                    status = await exchange.get_order_status(order_id)

                    batch_results.append(
                        {
                            "order_id": order_id,
                            "latency": latency,
                            "status": status,
                            "success": True,
                        }
                    )

                except Exception as e:
                    batch_results.append(
                        {"error": str(e), "latency": time.time() - start_time, "success": False}
                    )

            return {
                "batch_id": batch_id,
                "results": batch_results,
                "success_count": sum(1 for r in batch_results if r["success"]),
                "avg_latency": statistics.mean(r["latency"] for r in batch_results),
            }

        # Process orders in batches
        batch_tasks = [process_order_batch(i, batch_size) for i in range(order_count // batch_size)]

        batch_results = await asyncio.gather(*batch_tasks)
        self.load_test_monitor.stop()

        # Analyze performance
        total_orders = sum(len(batch["results"]) for batch in batch_results)
        total_successful = sum(batch["success_count"] for batch in batch_results)
        all_latencies = []

        for batch in batch_results:
            all_latencies.extend(r["latency"] for r in batch["results"])

        success_rate = total_successful / total_orders
        avg_latency = statistics.mean(all_latencies)
        p95_latency = statistics.quantiles(all_latencies, n=20)[18]  # 95th percentile
        throughput = total_orders / self.load_test_monitor.metrics["total_duration"]

        # Performance assertions
        assert success_rate >= 0.95  # 95% success rate
        assert avg_latency < 0.2  # Average latency under 200ms
        assert p95_latency < 0.5  # 95th percentile under 500ms
        assert throughput > 10  # At least 10 orders per second

        logger.info("High-frequency order processing results:")
        logger.info(f"  Orders processed: {total_orders}")
        logger.info(f"  Success rate: {success_rate:.1%}")
        logger.info(f"  Average latency: {avg_latency:.3f}s")
        logger.info(f"  95th percentile latency: {p95_latency:.3f}s")
        logger.info(f"  Throughput: {throughput:.1f} orders/sec")

        logger.info("✅ High-frequency order processing test passed")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_memory_usage_under_load(self):
        """Test memory usage characteristics under sustained load."""
        logger.info("Testing memory usage under load")

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create data structures that simulate real usage
        market_data_cache = []
        strategy_signals = []
        order_history = []

        async def generate_market_data(duration_seconds: int = 30):
            """Generate continuous market data stream."""
            start_time = time.time()
            data_count = 0

            while time.time() - start_time < duration_seconds:
                # Simulate market data
                market_data = MarketData(
                    symbol="BTC/USDT",
                    price=Decimal("50000.0") + Decimal(str((data_count % 100) - 50)),
                    bid=Decimal("49995.0"),
                    ask=Decimal("50005.0"),
                    volume=Decimal("100.0"),
                    timestamp=datetime.now(timezone.utc),
                )

                market_data_cache.append(market_data)
                data_count += 1

                # Limit cache size to simulate real-world scenario
                if len(market_data_cache) > 1000:
                    market_data_cache.pop(0)  # Remove oldest

                await asyncio.sleep(0.01)  # 100 data points per second

            return data_count

        async def generate_strategy_signals(duration_seconds: int = 30):
            """Generate strategy signals continuously."""
            start_time = time.time()
            signal_count = 0

            while time.time() - start_time < duration_seconds:
                signal = Signal(
                    symbol="BTC/USDT",
                    direction=SignalDirection.BUY
                    if signal_count % 3 == 0
                    else SignalDirection.HOLD,
                    confidence=0.7 + (signal_count % 10) * 0.03,
                    price=Decimal("50000.0"),
                    timestamp=datetime.now(timezone.utc),
                )

                strategy_signals.append(signal)
                signal_count += 1

                # Limit signal history
                if len(strategy_signals) > 500:
                    strategy_signals.pop(0)

                await asyncio.sleep(0.1)  # 10 signals per second

            return signal_count

        async def simulate_order_processing(duration_seconds: int = 30):
            """Simulate order processing and history tracking."""
            start_time = time.time()
            order_count = 0
            exchange = MockExchangeFactory.create_binance_mock()

            while time.time() - start_time < duration_seconds:
                # Create mock order
                order = Order(
                    id=f"order_{order_count}",
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=Decimal("0.01"),
                    price=Decimal("50000.0"),
                    status=OrderStatus.FILLED,
                )

                order_history.append(order)
                order_count += 1

                # Limit order history
                if len(order_history) > 200:
                    order_history.pop(0)

                await asyncio.sleep(0.2)  # 5 orders per second

            return order_count

        # Run concurrent load generation
        memory_samples = []

        async def monitor_memory():
            """Monitor memory usage during load test."""
            while True:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(current_memory)
                await asyncio.sleep(1.0)  # Sample every second

        # Start memory monitoring
        memory_task = asyncio.create_task(monitor_memory())

        try:
            # Run load generators concurrently
            results = await asyncio.gather(
                generate_market_data(30),
                generate_strategy_signals(30),
                simulate_order_processing(30),
            )

            data_points, signals, orders = results

        finally:
            memory_task.cancel()
            try:
                await memory_task
            except asyncio.CancelledError:
                pass

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_samples) if memory_samples else final_memory
        avg_memory = statistics.mean(memory_samples) if memory_samples else final_memory

        # Performance assertions
        assert memory_growth < 100  # Memory growth under 100MB
        assert max_memory < initial_memory + 150  # Peak memory under 150MB above initial

        # Log memory metrics
        logger.info("Memory usage analysis:")
        logger.info(f"  Initial memory: {initial_memory:.1f} MB")
        logger.info(f"  Final memory: {final_memory:.1f} MB")
        logger.info(f"  Memory growth: {memory_growth:.1f} MB")
        logger.info(f"  Peak memory: {max_memory:.1f} MB")
        logger.info(f"  Average memory: {avg_memory:.1f} MB")
        logger.info(
            f"  Data processed: {data_points} market data, {signals} signals, {orders} orders"
        )

        logger.info("✅ Memory usage under load test passed")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_scalable_strategy_execution(self):
        """Test strategy execution scalability with multiple concurrent strategies."""
        logger.info("Testing scalable strategy execution")

        # Create multiple strategy instances
        strategies = {
            f"momentum_{i}": MockStrategyFactory.create_momentum_strategy(
                signal_frequency=0.3 + (i * 0.1)
            )
            for i in range(5)
        }

        strategies.update(
            {
                f"mean_reversion_{i}": MockStrategyFactory.create_mean_reversion_strategy(
                    signal_confidence=0.6 + (i * 0.05)
                )
                for i in range(5)
            }
        )

        self.load_test_monitor.start()

        async def run_strategy_cycle(strategy_name: str, strategy: Mock, cycles: int = 50):
            """Run strategy through multiple execution cycles."""
            signals_generated = []
            execution_times = []
            errors = 0

            for cycle in range(cycles):
                start_time = time.time()
                try:
                    signal = await strategy.generate_signal()
                    execution_time = time.time() - start_time

                    signals_generated.append(signal)
                    execution_times.append(execution_time)

                    # Simulate small delay between cycles
                    await asyncio.sleep(0.01)

                except Exception as e:
                    errors += 1
                    logger.error(f"Strategy {strategy_name} error in cycle {cycle}: {e}")

            return {
                "strategy_name": strategy_name,
                "cycles_completed": len(signals_generated),
                "signals_generated": len(
                    [s for s in signals_generated if s.direction != SignalDirection.HOLD]
                ),
                "avg_execution_time": statistics.mean(execution_times) if execution_times else 0,
                "max_execution_time": max(execution_times) if execution_times else 0,
                "errors": errors,
            }

        # Run all strategies concurrently
        strategy_tasks = [
            run_strategy_cycle(name, strategy) for name, strategy in strategies.items()
        ]

        strategy_results = await asyncio.gather(*strategy_tasks)
        self.load_test_monitor.stop()

        # Analyze scalability results
        total_cycles = sum(r["cycles_completed"] for r in strategy_results)
        total_signals = sum(r["signals_generated"] for r in strategy_results)
        total_errors = sum(r["errors"] for r in strategy_results)

        all_execution_times = []
        for result in strategy_results:
            if result["avg_execution_time"] > 0:
                all_execution_times.append(result["avg_execution_time"])

        avg_strategy_execution_time = (
            statistics.mean(all_execution_times) if all_execution_times else 0
        )
        total_duration = self.load_test_monitor.metrics["total_duration"]
        strategy_throughput = total_cycles / total_duration

        # Scalability assertions
        assert total_errors < total_cycles * 0.02  # Error rate under 2%
        assert avg_strategy_execution_time < 0.1  # Average execution under 100ms
        assert strategy_throughput > 50  # At least 50 strategy cycles per second

        logger.info("Strategy execution scalability results:")
        logger.info(f"  Strategies: {len(strategies)}")
        logger.info(f"  Total cycles: {total_cycles}")
        logger.info(f"  Total signals: {total_signals}")
        logger.info(f"  Error rate: {total_errors / total_cycles:.1%}")
        logger.info(f"  Average execution time: {avg_strategy_execution_time:.3f}s")
        logger.info(f"  Strategy throughput: {strategy_throughput:.1f} cycles/sec")

        logger.info("✅ Scalable strategy execution test passed")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_database_connection_pool_performance(self):
        """Test database connection pool performance under concurrent load."""
        logger.info("Testing database connection pool performance")

        # Mock database connection pool
        class MockDatabasePool:
            def __init__(self, pool_size: int = 10):
                self.pool_size = pool_size
                self.active_connections = 0
                self.total_requests = 0
                self.wait_times = []

            async def execute_query(self, query: str):
                """Simulate database query execution."""
                self.total_requests += 1

                # Simulate connection acquisition wait time
                wait_start = time.time()
                while self.active_connections >= self.pool_size:
                    await asyncio.sleep(0.001)  # Wait for connection

                wait_time = time.time() - wait_start
                self.wait_times.append(wait_time)

                self.active_connections += 1

                try:
                    # Simulate query execution time
                    await asyncio.sleep(0.01)  # 10ms query time
                    return {"result": "success", "query": query}
                finally:
                    self.active_connections -= 1

        db_pool = MockDatabasePool(pool_size=20)

        async def concurrent_database_operations(client_id: int, operations: int = 30):
            """Perform concurrent database operations."""
            results = []

            for op in range(operations):
                query = f"SELECT * FROM trades WHERE client_id = {client_id} AND op = {op}"
                start_time = time.time()

                try:
                    result = await db_pool.execute_query(query)
                    execution_time = time.time() - start_time

                    results.append(
                        {"success": True, "execution_time": execution_time, "operation": op}
                    )

                except Exception as e:
                    results.append({"success": False, "error": str(e), "operation": op})

            return {
                "client_id": client_id,
                "operations": len(results),
                "successful": sum(1 for r in results if r["success"]),
                "avg_execution_time": statistics.mean(
                    r["execution_time"] for r in results if r.get("execution_time")
                )
                if results
                else 0,
            }

        # Run concurrent database clients
        client_count = 15
        client_tasks = [
            concurrent_database_operations(client_id) for client_id in range(client_count)
        ]

        self.load_test_monitor.start()
        client_results = await asyncio.gather(*client_tasks)
        self.load_test_monitor.stop()

        # Analyze database performance
        total_operations = sum(r["operations"] for r in client_results)
        total_successful = sum(r["successful"] for r in client_results)
        success_rate = total_successful / total_operations

        avg_wait_time = statistics.mean(db_pool.wait_times) if db_pool.wait_times else 0
        max_wait_time = max(db_pool.wait_times) if db_pool.wait_times else 0

        operations_per_second = total_operations / self.load_test_monitor.metrics["total_duration"]

        # Performance assertions
        assert success_rate >= 0.99  # 99% success rate
        assert avg_wait_time < 0.1  # Average wait time under 100ms
        assert max_wait_time < 0.5  # Max wait time under 500ms
        assert operations_per_second > 100  # At least 100 operations per second

        logger.info("Database connection pool performance:")
        logger.info(f"  Pool size: {db_pool.pool_size}")
        logger.info(f"  Concurrent clients: {client_count}")
        logger.info(f"  Total operations: {total_operations}")
        logger.info(f"  Success rate: {success_rate:.1%}")
        logger.info(f"  Average wait time: {avg_wait_time:.3f}s")
        logger.info(f"  Max wait time: {max_wait_time:.3f}s")
        logger.info(f"  Operations/sec: {operations_per_second:.1f}")

        logger.info("✅ Database connection pool performance test passed")

    async def run_integration_test(self):
        """Run all performance and scalability integration tests."""
        logger.info("Starting performance and scalability integration tests")

        test_methods = [
            self.test_concurrent_exchange_operations,
            self.test_high_frequency_order_processing,
            self.test_memory_usage_under_load,
            self.test_scalable_strategy_execution,
            self.test_database_connection_pool_performance,
        ]

        results = {}
        for test_method in test_methods:
            test_name = test_method.__name__
            try:
                logger.info(f"Running {test_name}")
                await test_method()
                results[test_name] = {"status": "PASSED"}
                logger.info(f"✅ {test_name} PASSED")
            except Exception as e:
                results[test_name] = {"status": "FAILED", "error": str(e)}
                logger.error(f"❌ {test_name} FAILED: {e}")
                # Continue with other tests

        # Summary
        passed = sum(1 for r in results.values() if r["status"] == "PASSED")
        total = len(results)

        logger.info(f"Performance and scalability test summary: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 All performance and scalability tests PASSED!")
        else:
            logger.error(f"💥 {total - passed} performance and scalability tests FAILED!")

        return results


@pytest_asyncio.fixture
async def performance_test():
    """Create performance and scalability test instance."""
    test = PerformanceScalabilityTest()
    await test.setup_integration_test()
    return test


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_comprehensive_performance_scalability():
    """Main test entry point for performance and scalability."""
    test = PerformanceScalabilityTest()
    results = await test.run_integration_test()

    # Verify critical performance tests passed
    critical_tests = [
        "test_concurrent_exchange_operations",
        "test_high_frequency_order_processing",
        "test_memory_usage_under_load",
    ]

    failed_critical = []
    for test_name in critical_tests:
        if results.get(test_name, {}).get("status") != "PASSED":
            failed_critical.append(test_name)

    assert len(failed_critical) == 0, f"Critical performance tests failed: {failed_critical}"


if __name__ == "__main__":
    # Run tests directly
    async def main():
        test = PerformanceScalabilityTest()
        await test.run_integration_test()

    asyncio.run(main())
