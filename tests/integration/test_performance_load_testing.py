"""
Performance and Load Testing Integration Tests

This module contains comprehensive performance and load testing scenarios for the T-Bot trading system.
Tests validate system behavior under various load conditions, measure performance metrics,
and ensure the system can handle production-level traffic.
"""

import asyncio
import logging
import statistics
import time
from decimal import Decimal

import pytest

from src.core.types import OrderSide, OrderType
from src.core.types.market import MarketData
from src.core.types.trading import Order
from tests.integration.base_integration import BaseIntegrationTest


class LoadTestMetrics:
    """Tracks comprehensive metrics during load testing."""

    def __init__(self):
        self.request_times: list[float] = []
        self.success_count = 0
        self.error_count = 0
        self.timeout_count = 0
        self.memory_samples: list[float] = []
        self.cpu_samples: list[float] = []
        self.concurrent_connections = 0
        self.throughput_samples: list[float] = []

    def add_request_time(self, duration: float):
        """Add request duration measurement."""
        self.request_times.append(duration)

    def add_success(self):
        """Increment success counter."""
        self.success_count += 1

    def add_error(self):
        """Increment error counter."""
        self.error_count += 1

    def add_timeout(self):
        """Increment timeout counter."""
        self.timeout_count += 1

    def get_statistics(self) -> dict:
        """Calculate comprehensive performance statistics."""
        if not self.request_times:
            return {}

        return {
            "total_requests": len(self.request_times),
            "success_rate": self.success_count / (self.success_count + self.error_count)
            if (self.success_count + self.error_count) > 0
            else 0,
            "avg_response_time": statistics.mean(self.request_times),
            "median_response_time": statistics.median(self.request_times),
            "p95_response_time": sorted(self.request_times)[int(len(self.request_times) * 0.95)]
            if len(self.request_times) > 0
            else 0,
            "p99_response_time": sorted(self.request_times)[int(len(self.request_times) * 0.99)]
            if len(self.request_times) > 0
            else 0,
            "min_response_time": min(self.request_times),
            "max_response_time": max(self.request_times),
            "errors_per_second": self.error_count / max(sum(self.request_times), 1),
            "timeouts": self.timeout_count,
            "throughput_rps": len(self.request_times) / max(sum(self.request_times), 1)
            if self.request_times
            else 0,
        }


class TestHighFrequencyTradingLoad(BaseIntegrationTest):
    """Test system performance under high-frequency trading scenarios."""

    async def setup_method(self):
        """Setup for high-frequency trading tests."""
        await super().setup_method()
        self.load_metrics = LoadTestMetrics()

        # Setup high-frequency trading configuration
        self.hft_config = {
            "max_orders_per_second": 100,
            "max_concurrent_orders": 50,
            "latency_threshold_ms": 10,
            "throughput_threshold_tps": 500,
        }

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_burst_order_processing(self):
        """Test system handling of burst order processing."""
        logger = logging.getLogger(__name__)
        logger.info("Testing burst order processing performance")

        # Create burst of orders
        orders = []
        for i in range(100):
            order = Order(
                id=f"burst_order_{i}",
                symbol="BTC/USDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                amount=Decimal("0.01"),
                price=Decimal(f"{50000 + (i * 10)}"),
                type=OrderType.LIMIT,
                exchange="binance",
            )
            orders.append(order)

        # Execute burst processing
        start_time = time.time()

        async def process_order(order: Order) -> bool:
            """Process single order and measure performance."""
            order_start = time.time()
            try:
                # Simulate order processing with realistic delay
                await asyncio.sleep(0.001)  # 1ms processing time

                # Mock exchange submission
                result = await self._mock_exchange_submit_order(order)

                duration = time.time() - order_start
                self.load_metrics.add_request_time(duration)

                if result:
                    self.load_metrics.add_success()
                    return True
                else:
                    self.load_metrics.add_error()
                    return False

            except asyncio.TimeoutError:
                self.load_metrics.add_timeout()
                return False
            except Exception as e:
                logger.error(f"Order processing error: {e}")
                self.load_metrics.add_error()
                return False

        # Process orders concurrently
        results = await asyncio.gather(
            *[process_order(order) for order in orders], return_exceptions=True
        )

        total_duration = time.time() - start_time

        # Calculate performance metrics
        stats = self.load_metrics.get_statistics()
        successful_orders = sum(1 for r in results if r is True)

        # Performance assertions
        assert stats["success_rate"] >= 0.95, (
            f"Success rate {stats['success_rate']} below threshold"
        )
        assert stats["avg_response_time"] < 0.05, (
            f"Average response time {stats['avg_response_time']}s too high"
        )
        assert stats["p95_response_time"] < 0.1, (
            f"P95 response time {stats['p95_response_time']}s too high"
        )
        assert successful_orders >= 95, (
            f"Only {successful_orders}/100 orders processed successfully"
        )
        assert total_duration < 10.0, f"Total processing time {total_duration}s too long"

        logger.info(
            f"Burst test completed: {successful_orders}/100 orders in {total_duration:.2f}s"
        )

    async def _mock_exchange_submit_order(self, order: Order) -> bool:
        """Mock exchange order submission with realistic behavior."""
        # Simulate network latency and processing time
        latency = 0.002 + (0.001 * asyncio.get_event_loop().time() % 3)  # 2-5ms latency
        await asyncio.sleep(latency)

        # Simulate 99% success rate
        return asyncio.get_event_loop().time() % 100 > 1

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_concurrent_market_data_processing(self):
        """Test concurrent market data processing under load."""
        logger = logging.getLogger(__name__)
        logger.info("Testing concurrent market data processing")

        # Setup multiple market data streams
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "DOT/USDT"]
        messages_per_symbol = 200

        async def process_market_data_stream(symbol: str):
            """Process market data stream for a symbol."""
            for i in range(messages_per_symbol):
                start_time = time.time()

                # Create market data message
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=int(time.time() * 1000),
                    price=Decimal(f"{50000 + (i * 10)}"),
                    volume=Decimal("1.5"),
                    bid=Decimal(f"{49999 + (i * 10)}"),
                    ask=Decimal(f"{50001 + (i * 10)}"),
                )

                try:
                    # Simulate processing
                    await self._process_market_data(market_data)

                    duration = time.time() - start_time
                    self.load_metrics.add_request_time(duration)
                    self.load_metrics.add_success()

                except Exception as e:
                    logger.error(f"Market data processing error: {e}")
                    self.load_metrics.add_error()

                # Small delay to simulate realistic data flow
                await asyncio.sleep(0.001)

        # Process all streams concurrently
        start_time = time.time()
        await asyncio.gather(*[process_market_data_stream(symbol) for symbol in symbols])
        total_duration = time.time() - start_time

        # Calculate performance metrics
        stats = self.load_metrics.get_statistics()
        total_messages = len(symbols) * messages_per_symbol

        # Performance assertions
        assert stats["success_rate"] >= 0.98, (
            f"Market data success rate {stats['success_rate']} too low"
        )
        assert stats["avg_response_time"] < 0.01, (
            f"Average processing time {stats['avg_response_time']}s too high"
        )
        assert total_messages / total_duration >= 500, (
            f"Throughput {total_messages / total_duration} msg/s too low"
        )

        logger.info(f"Processed {total_messages} market data messages in {total_duration:.2f}s")

    async def _process_market_data(self, market_data: MarketData):
        """Mock market data processing."""
        # Simulate data validation and processing
        await asyncio.sleep(0.0005)  # 0.5ms processing time


class TestWebSocketConnectionLoad(BaseIntegrationTest):
    """Test WebSocket connection handling under load."""

    async def setup_method(self):
        """Setup WebSocket load testing."""
        await super().setup_method()
        self.connection_metrics = LoadTestMetrics()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_multiple_websocket_connections(self):
        """Test handling multiple concurrent WebSocket connections."""
        logger = logging.getLogger(__name__)
        logger.info("Testing multiple WebSocket connection handling")

        num_connections = 10
        messages_per_connection = 100

        async def simulate_websocket_connection(connection_id: int):
            """Simulate a WebSocket connection with message flow."""
            connection_start = time.time()

            try:
                # Simulate connection establishment
                await asyncio.sleep(0.1)  # Connection establishment time

                for i in range(messages_per_connection):
                    message_start = time.time()

                    # Simulate message processing
                    message = {
                        "type": "ticker",
                        "symbol": "BTC/USDT",
                        "price": 50000 + (i * 10),
                        "connection_id": connection_id,
                    }

                    await self._process_websocket_message(message)

                    duration = time.time() - message_start
                    self.connection_metrics.add_request_time(duration)
                    self.connection_metrics.add_success()

                    # Simulate realistic message frequency
                    await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"WebSocket connection {connection_id} error: {e}")
                self.connection_metrics.add_error()

        # Create multiple concurrent connections
        start_time = time.time()
        await asyncio.gather(*[simulate_websocket_connection(i) for i in range(num_connections)])
        total_duration = time.time() - start_time

        # Calculate metrics
        stats = self.connection_metrics.get_statistics()
        total_messages = num_connections * messages_per_connection

        # Performance assertions
        assert stats["success_rate"] >= 0.99, (
            f"WebSocket success rate {stats['success_rate']} too low"
        )
        assert stats["avg_response_time"] < 0.005, (
            f"Average message processing {stats['avg_response_time']}s too slow"
        )
        assert total_messages / total_duration >= 80, (
            f"Message throughput {total_messages / total_duration} msg/s too low"
        )

        logger.info(
            f"Handled {num_connections} connections with {total_messages} messages in {total_duration:.2f}s"
        )

    async def _process_websocket_message(self, message: dict):
        """Mock WebSocket message processing."""
        # Simulate message parsing and routing
        await asyncio.sleep(0.0002)  # 0.2ms processing time

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_websocket_connection_resilience(self):
        """Test WebSocket connection resilience under stress."""
        logger = logging.getLogger(__name__)
        logger.info("Testing WebSocket connection resilience")

        # Simulate connection drops and reconnections
        connection_drops = 0
        successful_reconnections = 0

        for cycle in range(5):
            try:
                # Simulate normal operation
                await asyncio.sleep(1.0)

                # Simulate connection drop
                if cycle % 2 == 0:  # Drop every other cycle
                    connection_drops += 1
                    logger.info(f"Simulating connection drop {connection_drops}")

                    # Simulate reconnection logic
                    reconnect_start = time.time()
                    await asyncio.sleep(0.5)  # Reconnection delay
                    reconnect_time = time.time() - reconnect_start

                    if reconnect_time < 2.0:  # Successful reconnection within 2s
                        successful_reconnections += 1
                        self.connection_metrics.add_success()
                    else:
                        self.connection_metrics.add_timeout()

            except Exception as e:
                logger.error(f"Connection resilience test error: {e}")
                self.connection_metrics.add_error()

        # Assertions for resilience
        reconnection_rate = (
            successful_reconnections / connection_drops if connection_drops > 0 else 0
        )
        assert reconnection_rate >= 0.8, f"Reconnection rate {reconnection_rate} too low"

        logger.info(
            f"Connection resilience test: {successful_reconnections}/{connection_drops} successful reconnections"
        )


class TestDatabasePerformanceLoad(BaseIntegrationTest):
    """Test database performance under load conditions."""

    async def setup_method(self):
        """Setup database performance testing."""
        await super().setup_method()
        self.db_metrics = LoadTestMetrics()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_concurrent_database_operations(self):
        """Test concurrent database read/write operations."""
        logger = logging.getLogger(__name__)
        logger.info("Testing concurrent database operations")

        num_concurrent_ops = 50
        operations_per_task = 20

        async def database_operation_task(task_id: int):
            """Simulate database operations."""
            for i in range(operations_per_task):
                op_start = time.time()

                try:
                    # Simulate mix of read/write operations
                    if i % 3 == 0:  # Write operation
                        await self._mock_database_write(f"task_{task_id}_op_{i}")
                    else:  # Read operation
                        await self._mock_database_read(f"task_{task_id}_op_{i}")

                    duration = time.time() - op_start
                    self.db_metrics.add_request_time(duration)
                    self.db_metrics.add_success()

                except Exception as e:
                    logger.error(f"Database operation error: {e}")
                    self.db_metrics.add_error()

        # Execute concurrent database operations
        start_time = time.time()
        await asyncio.gather(*[database_operation_task(i) for i in range(num_concurrent_ops)])
        total_duration = time.time() - start_time

        # Calculate performance metrics
        stats = self.db_metrics.get_statistics()
        total_operations = num_concurrent_ops * operations_per_task

        # Performance assertions
        assert stats["success_rate"] >= 0.95, (
            f"Database success rate {stats['success_rate']} too low"
        )
        assert stats["avg_response_time"] < 0.1, (
            f"Average database response {stats['avg_response_time']}s too slow"
        )
        assert stats["p95_response_time"] < 0.2, (
            f"P95 database response {stats['p95_response_time']}s too slow"
        )
        assert total_operations / total_duration >= 200, (
            f"Database throughput {total_operations / total_duration} ops/s too low"
        )

        logger.info(f"Database load test: {total_operations} operations in {total_duration:.2f}s")

    async def _mock_database_write(self, operation_id: str):
        """Mock database write operation."""
        # Simulate write latency
        await asyncio.sleep(0.005 + (0.002 * (hash(operation_id) % 3)))

    async def _mock_database_read(self, operation_id: str):
        """Mock database read operation."""
        # Simulate read latency (faster than writes)
        await asyncio.sleep(0.002 + (0.001 * (hash(operation_id) % 3)))

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_database_connection_pool_performance(self):
        """Test database connection pool under load."""
        logger = logging.getLogger(__name__)
        logger.info("Testing database connection pool performance")

        # Simulate connection pool behavior
        pool_size = 10
        active_connections = 0
        max_active_connections = 0
        connection_wait_times = []

        async def simulate_database_query(query_id: int):
            """Simulate database query with connection pool."""
            nonlocal active_connections, max_active_connections

            # Wait for available connection
            wait_start = time.time()
            while active_connections >= pool_size:
                await asyncio.sleep(0.001)
            wait_time = time.time() - wait_start
            connection_wait_times.append(wait_time)

            # Acquire connection
            active_connections += 1
            max_active_connections = max(max_active_connections, active_connections)

            try:
                # Simulate query execution
                query_start = time.time()
                await asyncio.sleep(0.01 + (0.005 * (query_id % 3)))  # Variable query time
                query_duration = time.time() - query_start

                self.db_metrics.add_request_time(query_duration)
                self.db_metrics.add_success()

            except Exception as e:
                logger.error(f"Database query {query_id} error: {e}")
                self.db_metrics.add_error()
            finally:
                # Release connection
                active_connections -= 1

        # Execute concurrent queries
        num_queries = 100
        start_time = time.time()
        await asyncio.gather(*[simulate_database_query(i) for i in range(num_queries)])
        total_duration = time.time() - start_time

        # Calculate connection pool metrics
        avg_wait_time = statistics.mean(connection_wait_times) if connection_wait_times else 0
        max_wait_time = max(connection_wait_times) if connection_wait_times else 0

        # Performance assertions
        assert avg_wait_time < 0.05, f"Average connection wait time {avg_wait_time}s too high"
        assert max_wait_time < 0.2, f"Maximum connection wait time {max_wait_time}s too high"
        assert max_active_connections <= pool_size, (
            f"Connection pool overflow: {max_active_connections} > {pool_size}"
        )

        stats = self.db_metrics.get_statistics()
        assert stats["success_rate"] >= 0.98, (
            f"Database query success rate {stats['success_rate']} too low"
        )

        logger.info(
            f"Connection pool test: {num_queries} queries, avg wait: {avg_wait_time:.3f}s, max active: {max_active_connections}"
        )


class TestMemoryAndResourceLoad(BaseIntegrationTest):
    """Test memory usage and resource management under load."""

    async def setup_method(self):
        """Setup memory and resource testing."""
        await super().setup_method()
        self.resource_metrics = LoadTestMetrics()
        self.memory_snapshots = []

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_memory_usage_under_load(self):
        """Test memory usage patterns under sustained load."""
        logger = logging.getLogger(__name__)
        logger.info("Testing memory usage under sustained load")

        import gc

        import psutil

        # Baseline memory usage
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_snapshots.append(initial_memory)

        # Generate sustained load
        large_data_structures = []

        for cycle in range(10):
            cycle_start = time.time()

            try:
                # Create memory-intensive operations
                for i in range(100):
                    # Simulate large market data structures
                    market_data_batch = [
                        {
                            "symbol": f"PAIR_{j}",
                            "price": 1000 + (i * 10) + j,
                            "volume": 1.5 + (j * 0.1),
                            "timestamp": int(time.time() * 1000),
                            "bid": 999 + (i * 10) + j,
                            "ask": 1001 + (i * 10) + j,
                        }
                        for j in range(50)
                    ]
                    large_data_structures.append(market_data_batch)

                    # Process data
                    await self._process_market_data_batch(market_data_batch)

                # Measure memory after each cycle
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                self.memory_snapshots.append(current_memory)

                # Clean up some data structures to simulate garbage collection
                if len(large_data_structures) > 500:
                    large_data_structures = large_data_structures[-250:]
                    gc.collect()

                cycle_duration = time.time() - cycle_start
                self.resource_metrics.add_request_time(cycle_duration)
                self.resource_metrics.add_success()

                logger.info(
                    f"Memory cycle {cycle}: {current_memory:.1f}MB (delta: {current_memory - initial_memory:.1f}MB)"
                )

            except Exception as e:
                logger.error(f"Memory test cycle {cycle} error: {e}")
                self.resource_metrics.add_error()

        # Final memory measurement
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_snapshots.append(final_memory)

        # Memory usage analysis
        memory_growth = final_memory - initial_memory
        max_memory = max(self.memory_snapshots)
        avg_memory = statistics.mean(self.memory_snapshots)

        # Memory assertions (adjust thresholds based on system requirements)
        assert memory_growth < 500, (
            f"Memory growth {memory_growth:.1f}MB exceeds threshold"
        )  # 500MB max growth
        assert max_memory < initial_memory + 1000, (
            f"Peak memory {max_memory:.1f}MB too high"
        )  # 1GB max peak

        logger.info(
            f"Memory test completed: Growth {memory_growth:.1f}MB, Peak {max_memory:.1f}MB, Avg {avg_memory:.1f}MB"
        )

    async def _process_market_data_batch(self, batch: list[dict]):
        """Mock processing of market data batch."""
        # Simulate CPU-intensive processing
        await asyncio.sleep(0.001)

        # Simulate some calculations
        total_volume = sum(item["volume"] for item in batch)
        avg_price = statistics.mean(item["price"] for item in batch)

        # Return processed results (simulation)
        return {"total_volume": total_volume, "avg_price": avg_price, "batch_size": len(batch)}

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_resource_cleanup_efficiency(self):
        """Test resource cleanup and garbage collection efficiency."""
        logger = logging.getLogger(__name__)
        logger.info("Testing resource cleanup efficiency")

        import gc
        import weakref

        # Track object lifecycle
        created_objects = []
        weak_references = []

        class MockTradingObject:
            """Mock trading object for lifecycle testing."""

            def __init__(self, object_id: str):
                self.id = object_id
                self.data = [i for i in range(1000)]  # Some memory footprint

            def cleanup(self):
                """Mock cleanup method."""
                self.data = None

        # Create and track objects
        for i in range(100):
            obj = MockTradingObject(f"obj_{i}")
            created_objects.append(obj)
            weak_references.append(weakref.ref(obj))

        # Simulate usage and cleanup
        active_objects = len(created_objects)
        cleanup_start = time.time()

        # Clean up objects in batches
        for batch_start in range(0, len(created_objects), 20):
            batch_end = min(batch_start + 20, len(created_objects))

            # Cleanup batch
            for i in range(batch_start, batch_end):
                if i < len(created_objects):
                    created_objects[i].cleanup()
                    created_objects[i] = None

            # Force garbage collection
            gc.collect()

            # Check weak references to see if objects were actually cleaned up
            alive_refs = sum(1 for ref in weak_references if ref() is not None)
            logger.info(f"After batch cleanup: {alive_refs} objects still alive")

            await asyncio.sleep(0.01)  # Small delay to simulate real-world usage

        cleanup_duration = time.time() - cleanup_start

        # Final garbage collection
        created_objects.clear()
        gc.collect()

        # Count remaining objects
        final_alive_refs = sum(1 for ref in weak_references if ref() is not None)
        cleanup_efficiency = (100 - final_alive_refs) / 100.0

        # Resource cleanup assertions
        assert cleanup_efficiency >= 0.95, f"Cleanup efficiency {cleanup_efficiency} too low"
        assert cleanup_duration < 5.0, f"Cleanup duration {cleanup_duration}s too long"
        assert final_alive_refs <= 5, f"Too many objects ({final_alive_refs}) not cleaned up"

        logger.info(
            f"Resource cleanup test: {cleanup_efficiency:.2%} efficiency in {cleanup_duration:.2f}s"
        )


class TestSystemIntegrationLoad(BaseIntegrationTest):
    """Test full system integration under load conditions."""

    async def setup_method(self):
        """Setup system integration load testing."""
        await super().setup_method()
        self.system_metrics = LoadTestMetrics()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_end_to_end_system_load(self):
        """Test complete system under realistic production load."""
        logger = logging.getLogger(__name__)
        logger.info("Testing end-to-end system under load")

        # Define load test parameters
        load_test_config = {
            "concurrent_users": 20,
            "test_duration_seconds": 30,
            "orders_per_user_per_second": 2,
            "market_data_updates_per_second": 100,
            "websocket_connections": 10,
        }

        # Track system-wide metrics
        total_orders_processed = 0
        total_market_updates = 0
        total_websocket_messages = 0

        async def simulate_trading_user(user_id: int):
            """Simulate a trading user generating load."""
            nonlocal total_orders_processed

            orders_processed = 0
            end_time = time.time() + load_test_config["test_duration_seconds"]

            while time.time() < end_time:
                try:
                    # Create and submit order
                    order = Order(
                        id=f"load_order_{user_id}_{orders_processed}",
                        symbol="BTC/USDT",
                        side=OrderSide.BUY if orders_processed % 2 == 0 else OrderSide.SELL,
                        amount=Decimal("0.01"),
                        price=Decimal(f"{50000 + (orders_processed * 10)}"),
                        type=OrderType.LIMIT,
                        exchange="binance",
                    )

                    # Simulate order processing pipeline
                    start_time = time.time()
                    await self._process_order_pipeline(order)
                    duration = time.time() - start_time

                    self.system_metrics.add_request_time(duration)
                    self.system_metrics.add_success()
                    orders_processed += 1
                    total_orders_processed += 1

                    # Maintain target rate
                    await asyncio.sleep(1.0 / load_test_config["orders_per_user_per_second"])

                except Exception as e:
                    logger.error(f"Trading user {user_id} error: {e}")
                    self.system_metrics.add_error()

        async def simulate_market_data_feed():
            """Simulate market data feed."""
            nonlocal total_market_updates

            end_time = time.time() + load_test_config["test_duration_seconds"]

            while time.time() < end_time:
                try:
                    # Generate market data update
                    market_data = MarketData(
                        symbol="BTC/USDT",
                        timestamp=int(time.time() * 1000),
                        price=Decimal(f"{50000 + (total_market_updates % 1000)}"),
                        volume=Decimal("1.5"),
                        bid=Decimal(f"{49999 + (total_market_updates % 1000)}"),
                        ask=Decimal(f"{50001 + (total_market_updates % 1000)}"),
                    )

                    await self._process_market_data_update(market_data)
                    total_market_updates += 1

                    # Maintain target rate
                    await asyncio.sleep(1.0 / load_test_config["market_data_updates_per_second"])

                except Exception as e:
                    logger.error(f"Market data feed error: {e}")

        async def simulate_websocket_activity():
            """Simulate WebSocket message activity."""
            nonlocal total_websocket_messages

            end_time = time.time() + load_test_config["test_duration_seconds"]

            while time.time() < end_time:
                try:
                    # Generate WebSocket messages
                    for conn_id in range(load_test_config["websocket_connections"]):
                        message = {
                            "type": "price_update",
                            "symbol": "BTC/USDT",
                            "price": 50000 + (total_websocket_messages % 1000),
                            "connection_id": conn_id,
                        }

                        await self._process_websocket_message(message)
                        total_websocket_messages += 1

                    await asyncio.sleep(0.1)  # 10 messages per second per connection

                except Exception as e:
                    logger.error(f"WebSocket activity error: {e}")

        # Execute load test with all components
        start_time = time.time()

        # Run all load generators concurrently
        await asyncio.gather(
            # Trading users
            *[simulate_trading_user(i) for i in range(load_test_config["concurrent_users"])],
            # Market data feed
            simulate_market_data_feed(),
            # WebSocket activity
            simulate_websocket_activity(),
            return_exceptions=True,
        )

        total_duration = time.time() - start_time

        # Calculate comprehensive system metrics
        stats = self.system_metrics.get_statistics()

        # System performance assertions
        assert stats["success_rate"] >= 0.95, f"System success rate {stats['success_rate']} too low"
        assert stats["avg_response_time"] < 0.1, (
            f"System response time {stats['avg_response_time']}s too high"
        )
        assert total_orders_processed >= (
            load_test_config["concurrent_users"]
            * load_test_config["orders_per_user_per_second"]
            * load_test_config["test_duration_seconds"]
            * 0.8
        ), "Order processing rate too low"
        assert total_market_updates >= (
            load_test_config["market_data_updates_per_second"]
            * load_test_config["test_duration_seconds"]
            * 0.8
        ), "Market data processing rate too low"

        # Log final system performance summary
        logger.info(f"""
        System Load Test Results:
        - Duration: {total_duration:.2f}s
        - Orders Processed: {total_orders_processed}
        - Market Updates: {total_market_updates}  
        - WebSocket Messages: {total_websocket_messages}
        - Success Rate: {stats["success_rate"]:.2%}
        - Avg Response Time: {stats["avg_response_time"]:.3f}s
        - P95 Response Time: {stats["p95_response_time"]:.3f}s
        """)

    async def _process_order_pipeline(self, order: Order):
        """Mock complete order processing pipeline."""
        # Simulate order validation
        await asyncio.sleep(0.001)

        # Simulate risk checks
        await asyncio.sleep(0.002)

        # Simulate exchange submission
        await asyncio.sleep(0.005)

        # Simulate confirmation processing
        await asyncio.sleep(0.001)

    async def _process_market_data_update(self, market_data: MarketData):
        """Mock market data processing."""
        await asyncio.sleep(0.0005)

    async def _process_websocket_message(self, message: dict):
        """Mock WebSocket message processing."""
        await asyncio.sleep(0.0002)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_system_degradation_under_extreme_load(self):
        """Test system behavior under extreme load conditions."""
        logger = logging.getLogger(__name__)
        logger.info("Testing system degradation under extreme load")

        # Gradually increase load and measure degradation
        load_levels = [10, 50, 100, 200, 500]  # requests per second
        degradation_metrics = []

        for load_level in load_levels:
            logger.info(f"Testing load level: {load_level} rps")
            level_metrics = LoadTestMetrics()

            # Generate load for this level
            test_duration = 10  # seconds
            total_requests = load_level * test_duration
            request_interval = 1.0 / load_level

            async def generate_request(request_id: int):
                """Generate individual request."""
                start_time = time.time()
                try:
                    # Simulate system processing under load
                    processing_time = 0.01 + (0.001 * (load_level / 100))  # Increases with load
                    await asyncio.sleep(processing_time)

                    duration = time.time() - start_time
                    level_metrics.add_request_time(duration)
                    level_metrics.add_success()

                except Exception as e:
                    logger.error(f"Request {request_id} failed: {e}")
                    level_metrics.add_error()

            # Execute requests with controlled rate
            start_time = time.time()
            tasks = []

            for i in range(total_requests):
                task = asyncio.create_task(generate_request(i))
                tasks.append(task)

                if i < total_requests - 1:  # Don't sleep after last request
                    await asyncio.sleep(request_interval)

            # Wait for all requests to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            actual_duration = time.time() - start_time

            # Calculate metrics for this load level
            stats = level_metrics.get_statistics()
            actual_rps = total_requests / actual_duration

            degradation_metrics.append(
                {
                    "target_rps": load_level,
                    "actual_rps": actual_rps,
                    "success_rate": stats["success_rate"],
                    "avg_response_time": stats["avg_response_time"],
                    "p95_response_time": stats["p95_response_time"],
                    "degradation_factor": actual_rps / load_level if load_level > 0 else 0,
                }
            )

            logger.info(
                f"Load level {load_level}: {actual_rps:.1f} actual rps, {stats['success_rate']:.2%} success"
            )

        # Analyze degradation patterns
        baseline_performance = degradation_metrics[0]

        for i, metrics in enumerate(degradation_metrics):
            load_level = load_levels[i]

            # Check that system doesn't completely fail under any load
            assert metrics["success_rate"] >= 0.5, (
                f"System failed at {load_level} rps: {metrics['success_rate']:.2%} success rate"
            )

            # Check that response times don't become unreasonable
            assert metrics["avg_response_time"] <= 1.0, (
                f"Response time too high at {load_level} rps: {metrics['avg_response_time']:.3f}s"
            )

            # Check that throughput degradation is gradual, not cliff-like
            if i > 0:
                prev_metrics = degradation_metrics[i - 1]
                throughput_drop = (
                    prev_metrics["actual_rps"] - metrics["actual_rps"]
                ) / prev_metrics["actual_rps"]
                assert throughput_drop <= 0.5, (
                    f"Throughput cliff detected between {load_levels[i - 1]} and {load_level} rps"
                )

        logger.info(
            "System degradation test completed - system shows graceful degradation under extreme load"
        )
