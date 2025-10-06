"""
Real-time data flow integration tests.

Tests WebSocket connections, market data processing pipeline, real-time risk monitoring,
performance monitoring integration, and data synchronization across components.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest

from src.core.types import (
    MarketData,
)
from tests.integration.base_integration import (
    performance_test,
)

logger = logging.getLogger(__name__)


class TestWebSocketConnections:
    """Test WebSocket connections across all exchanges."""

    @pytest.mark.asyncio
    @performance_test(max_duration=45.0)
    @pytest.mark.timeout(300)
    async def test_multi_exchange_websocket_stability(self, performance_monitor):
        """Test stability of WebSocket connections across multiple exchanges."""

        # Mock WebSocket clients for different exchanges
        ws_clients = {}
        connection_status = {}
        message_counts = {}

        class MockWebSocketManager:
            def __init__(self, exchange_name: str):
                self.exchange_name = exchange_name
                self.connected = False
                self.reconnect_count = 0
                self.message_buffer = []
                self.heartbeat_interval = 30.0
                self.last_heartbeat = None

            async def connect(self):
                """Simulate WebSocket connection."""
                await asyncio.sleep(0.1)  # Simulate connection delay
                self.connected = True
                self.last_heartbeat = time.time()
                connection_status[self.exchange_name] = "connected"
                message_counts[self.exchange_name] = 0
                logger.info(f"{self.exchange_name} WebSocket connected")

                # Start message simulation
                asyncio.create_task(self._simulate_message_stream())
                asyncio.create_task(self._heartbeat_monitor())

            async def disconnect(self):
                """Simulate WebSocket disconnection."""
                self.connected = False
                connection_status[self.exchange_name] = "disconnected"
                logger.info(f"{self.exchange_name} WebSocket disconnected")

            async def _simulate_message_stream(self):
                """Simulate incoming market data messages."""
                import random

                while self.connected:
                    # Generate mock market data
                    symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
                    symbol = random.choice(symbols)

                    market_data = {
                        "type": "ticker",
                        "symbol": symbol,
                        "price": str(50000 + random.uniform(-1000, 1000)),
                        "volume": str(random.uniform(100, 2000)),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "exchange": self.exchange_name,
                    }

                    self.message_buffer.append(market_data)
                    message_counts[self.exchange_name] += 1

                    # Record performance metrics
                    performance_monitor.record_api_call()

                    # Simulate different message frequencies for different exchanges
                    if self.exchange_name == "binance":
                        await asyncio.sleep(0.1)  # High frequency
                    elif self.exchange_name == "coinbase":
                        await asyncio.sleep(0.2)  # Medium frequency
                    else:
                        await asyncio.sleep(0.3)  # Lower frequency

            async def _heartbeat_monitor(self):
                """Monitor connection health with heartbeats."""
                while self.connected:
                    await asyncio.sleep(self.heartbeat_interval)

                    if self.connected:
                        self.last_heartbeat = time.time()
                        logger.debug(f"{self.exchange_name} heartbeat")

                        # Simulate occasional connection issues
                        if random.random() < 0.05:  # 5% chance of connection issue
                            logger.warning(f"{self.exchange_name} connection issue detected")
                            await self._reconnect()

            async def _reconnect(self):
                """Handle reconnection logic."""
                self.reconnect_count += 1
                connection_status[self.exchange_name] = "reconnecting"

                await self.disconnect()
                await asyncio.sleep(1.0)  # Wait before reconnecting
                await self.connect()

            def get_buffered_messages(self) -> list[dict]:
                """Get and clear message buffer."""
                messages = self.message_buffer.copy()
                self.message_buffer.clear()
                return messages

        # Create WebSocket clients for multiple exchanges
        exchanges = ["binance", "coinbase", "okx"]

        for exchange_name in exchanges:
            ws_manager = MockWebSocketManager(exchange_name)
            ws_clients[exchange_name] = ws_manager

        # Connect all WebSocket clients
        connection_tasks = []
        for exchange_name, ws_manager in ws_clients.items():
            task = asyncio.create_task(ws_manager.connect())
            connection_tasks.append(task)

        await asyncio.gather(*connection_tasks)

        # Verify all connections established
        for exchange_name in exchanges:
            assert connection_status[exchange_name] == "connected"
            assert ws_clients[exchange_name].connected is True

        logger.info(f"All {len(exchanges)} WebSocket connections established")

        # Run data collection for test period
        test_duration = 10.0  # 10 seconds
        start_time = time.time()

        while time.time() - start_time < test_duration:
            # Collect messages from all exchanges
            total_messages = 0

            for exchange_name, ws_manager in ws_clients.items():
                messages = ws_manager.get_buffered_messages()
                total_messages += len(messages)

                # Process messages (mock processing)
                for message in messages:
                    # Simulate market data processing
                    price = Decimal(message["price"])
                    market_data = MarketData(
                        symbol=message["symbol"],
                        timestamp=datetime.fromisoformat(
                            message["timestamp"].replace("Z", "+00:00")
                        ),
                        open=price,
                        high=price + Decimal("1.0"),
                        low=price - Decimal("1.0"),
                        close=price,
                        volume=Decimal(message["volume"]),
                        exchange=message.get("exchange", "test"),
                    )

                    # Mock downstream processing
                    assert market_data.close > 0
                    assert market_data.volume > 0

            await asyncio.sleep(0.5)  # Check every 500ms

        # Verify message flow
        total_received = sum(message_counts.values())
        assert total_received > 0

        # Check connection stability
        stable_connections = sum(
            1 for status in connection_status.values() if status == "connected"
        )
        assert stable_connections >= len(exchanges) * 0.8  # At least 80% should be stable

        # Check reconnection handling
        total_reconnects = sum(ws_manager.reconnect_count for ws_manager in ws_clients.values())
        logger.info(f"Total reconnections during test: {total_reconnects}")

        # Disconnect all clients
        disconnect_tasks = []
        for ws_manager in ws_clients.values():
            task = asyncio.create_task(ws_manager.disconnect())
            disconnect_tasks.append(task)

        await asyncio.gather(*disconnect_tasks)

        # Verify disconnection
        for exchange_name in exchanges:
            assert connection_status[exchange_name] == "disconnected"
            assert ws_clients[exchange_name].connected is False

        logger.info(f"WebSocket stability test completed: {total_received} messages received")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_websocket_failover_mechanisms(self):
        """Test WebSocket failover and backup connection mechanisms."""

        class FailoverWebSocketManager:
            def __init__(self, exchange_name: str):
                self.exchange_name = exchange_name
                self.primary_connected = False
                self.backup_connected = False
                self.current_connection = None
                self.failover_count = 0
                self.message_queue = []

            async def connect_primary(self):
                """Connect to primary WebSocket endpoint."""
                await asyncio.sleep(0.1)
                self.primary_connected = True
                self.current_connection = "primary"
                logger.info(f"{self.exchange_name} primary WebSocket connected")

            async def connect_backup(self):
                """Connect to backup WebSocket endpoint."""
                await asyncio.sleep(0.2)  # Backup might be slower
                self.backup_connected = True
                self.current_connection = "backup"
                logger.info(f"{self.exchange_name} backup WebSocket connected")

            async def simulate_primary_failure(self):
                """Simulate primary connection failure."""
                if self.primary_connected:
                    self.primary_connected = False
                    if self.current_connection == "primary":
                        self.current_connection = None
                        logger.warning(f"{self.exchange_name} primary connection failed")

                        # Trigger failover
                        await self.failover_to_backup()

            async def failover_to_backup(self):
                """Failover to backup connection."""
                self.failover_count += 1

                if not self.backup_connected:
                    await self.connect_backup()

                self.current_connection = "backup"
                logger.info(
                    f"{self.exchange_name} failed over to backup (failover #{self.failover_count})"
                )

            async def recover_primary(self):
                """Recover primary connection."""
                await self.connect_primary()

                # Switch back to primary if it's preferred
                if self.primary_connected:
                    self.current_connection = "primary"
                    logger.info(f"{self.exchange_name} recovered to primary connection")

            def is_connected(self) -> bool:
                """Check if any connection is active."""
                return self.current_connection is not None

            def get_active_connection(self) -> str:
                """Get the currently active connection type."""
                return self.current_connection

        # Test failover scenario
        exchange_name = "binance"
        ws_manager = FailoverWebSocketManager(exchange_name)

        # 1. Connect to primary
        await ws_manager.connect_primary()

        assert ws_manager.is_connected() is True
        assert ws_manager.get_active_connection() == "primary"
        assert ws_manager.failover_count == 0

        # 2. Simulate primary failure
        await ws_manager.simulate_primary_failure()

        assert ws_manager.is_connected() is True
        assert ws_manager.get_active_connection() == "backup"
        assert ws_manager.failover_count == 1

        # 3. Test recovery to primary
        await ws_manager.recover_primary()

        assert ws_manager.is_connected() is True
        assert ws_manager.get_active_connection() == "primary"

        # 4. Test multiple failovers
        for i in range(3):
            await ws_manager.simulate_primary_failure()
            await asyncio.sleep(0.1)
            await ws_manager.recover_primary()
            await asyncio.sleep(0.1)

        assert ws_manager.failover_count == 4  # Original + 3 additional
        assert ws_manager.is_connected() is True

        logger.info(f"Failover test completed with {ws_manager.failover_count} failovers")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_websocket_message_ordering_and_deduplication(self):
        """Test message ordering and deduplication in WebSocket streams."""

        class OrderedWebSocketManager:
            def __init__(self):
                self.received_messages = []
                self.processed_messages = []
                self.duplicate_count = 0
                self.out_of_order_count = 0
                self.message_ids_seen = set()

            async def receive_message(self, message: dict[str, Any]):
                """Receive and process a message with ordering/deduplication."""
                self.received_messages.append(message)

                # Check for duplicates
                message_id = message.get("id")
                if message_id in self.message_ids_seen:
                    self.duplicate_count += 1
                    logger.warning(f"Duplicate message detected: {message_id}")
                    return  # Skip processing duplicate

                self.message_ids_seen.add(message_id)

                # Check for ordering (simplified - based on sequence number)
                sequence = message.get("sequence", 0)
                if self.processed_messages:
                    last_sequence = self.processed_messages[-1].get("sequence", 0)
                    if sequence < last_sequence:
                        self.out_of_order_count += 1
                        logger.warning(f"Out-of-order message: {sequence} < {last_sequence}")

                self.processed_messages.append(message)

            def get_stats(self) -> dict[str, int]:
                """Get processing statistics."""
                return {
                    "received": len(self.received_messages),
                    "processed": len(self.processed_messages),
                    "duplicates": self.duplicate_count,
                    "out_of_order": self.out_of_order_count,
                }

        ws_manager = OrderedWebSocketManager()

        # Generate test messages with some duplicates and out-of-order
        test_messages = []

        # Normal sequence
        for i in range(1, 11):
            message = {
                "id": f"msg_{i}",
                "sequence": i,
                "type": "ticker",
                "symbol": "BTC/USDT",
                "price": str(50000 + i),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            test_messages.append(message)

        # Add some duplicates
        test_messages.append(test_messages[5].copy())  # Duplicate message 6
        test_messages.append(test_messages[8].copy())  # Duplicate message 9

        # Add some out-of-order messages
        out_of_order_msg = {
            "id": "msg_5.5",
            "sequence": 5,  # Lower sequence than expected
            "type": "ticker",
            "symbol": "BTC/USDT",
            "price": "50005.5",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        test_messages.append(out_of_order_msg)

        # Process all messages
        for message in test_messages:
            await ws_manager.receive_message(message)

        # Verify results
        stats = ws_manager.get_stats()

        assert stats["received"] == len(test_messages)  # All messages received
        assert stats["processed"] == len(test_messages) - 2  # Minus duplicates
        assert stats["duplicates"] == 2  # Two duplicate messages
        assert stats["out_of_order"] >= 1  # At least one out-of-order

        logger.info(f"Message ordering test stats: {stats}")

        # Verify processed messages are unique
        processed_ids = [msg["id"] for msg in ws_manager.processed_messages]
        assert len(processed_ids) == len(set(processed_ids))  # All unique

        logger.info("Message ordering and deduplication test completed")


class TestMarketDataPipeline:
    """Test market data processing pipeline integration."""

    @pytest.mark.asyncio
    @performance_test(max_duration=20.0)
    @pytest.mark.timeout(300)
    async def test_data_processing_pipeline_throughput(self, performance_monitor):
        """Test data processing pipeline throughput under load."""

        class MockDataProcessor:
            def __init__(self):
                self.processed_count = 0
                self.processing_times = []
                self.error_count = 0
                self.validation_failures = 0

            async def process_market_data(self, raw_data: dict[str, Any]) -> MarketData:
                """Process raw market data into structured format."""
                start_time = time.time()

                try:
                    # Simulate data validation
                    if not self._validate_data(raw_data):
                        self.validation_failures += 1
                        raise ValueError("Invalid market data format")

                    # Simulate data transformation
                    await asyncio.sleep(0.001)  # 1ms processing time

                    price = Decimal(str(raw_data["price"]))
                    market_data = MarketData(
                        symbol=raw_data["symbol"],
                        timestamp=datetime.fromisoformat(
                            raw_data["timestamp"].replace("Z", "+00:00")
                        ),
                        open=price,
                        high=price + Decimal("1.0"),
                        low=price - Decimal("1.0"),
                        close=price,
                        volume=Decimal(str(raw_data.get("volume", "1000"))),
                        exchange=raw_data.get("exchange", "test"),
                    )

                    self.processed_count += 1
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)

                    return market_data

                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Data processing error: {e}")
                    raise

            def _validate_data(self, data: dict[str, Any]) -> bool:
                """Validate raw market data."""
                required_fields = ["symbol", "price", "timestamp"]

                for field in required_fields:
                    if field not in data:
                        return False

                # Price validation
                try:
                    price = float(data["price"])
                    if price <= 0:
                        return False
                except (ValueError, TypeError):
                    return False

                # Volume validation if present
                if "volume" in data:
                    try:
                        volume = float(data["volume"])
                        if volume < 0:
                            return False
                    except (ValueError, TypeError):
                        return False

                return True

            def get_performance_stats(self) -> dict[str, Any]:
                """Get processing performance statistics."""
                if self.processing_times:
                    avg_time = sum(self.processing_times) / len(self.processing_times)
                    max_time = max(self.processing_times)
                    min_time = min(self.processing_times)
                else:
                    avg_time = max_time = min_time = 0

                return {
                    "processed_count": self.processed_count,
                    "error_count": self.error_count,
                    "validation_failures": self.validation_failures,
                    "avg_processing_time": avg_time,
                    "max_processing_time": max_time,
                    "min_processing_time": min_time,
                    "throughput_per_second": self.processed_count / sum(self.processing_times)
                    if self.processing_times
                    else 0,
                }

        processor = MockDataProcessor()

        # Generate test data load
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]
        test_data_batch = []

        # Generate high-frequency data
        for i in range(1000):  # 1000 market data points
            import random

            symbol = random.choice(symbols)

            raw_data = {
                "symbol": symbol,
                "price": 50000 + random.uniform(-5000, 5000),
                "volume": random.uniform(100, 2000),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bid": None,  # Will be calculated
                "ask": None,  # Will be calculated
            }

            # Add some invalid data to test error handling
            if i % 100 == 0:  # Every 100th item
                raw_data["price"] = "invalid_price"  # Invalid price
            elif i % 150 == 0:  # Every 150th item
                del raw_data["symbol"]  # Missing required field

            test_data_batch.append(raw_data)

        # Process data batch
        processed_data = []
        processing_start = time.time()

        # Process in batches for better performance measurement
        batch_size = 50
        for i in range(0, len(test_data_batch), batch_size):
            batch = test_data_batch[i : i + batch_size]

            batch_tasks = []
            for raw_data in batch:
                task = asyncio.create_task(processor.process_market_data(raw_data))
                batch_tasks.append(task)

            # Process batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, MarketData):
                    processed_data.append(result)
                    performance_monitor.record_api_call()
                # Exceptions are already counted in processor error stats

        processing_end = time.time()
        total_processing_time = processing_end - processing_start

        # Analyze performance
        stats = processor.get_performance_stats()

        logger.info("Data processing pipeline performance:")
        logger.info(f"  Total processing time: {total_processing_time:.2f}s")
        logger.info(f"  Processed count: {stats['processed_count']}")
        logger.info(f"  Error count: {stats['error_count']}")
        logger.info(f"  Validation failures: {stats['validation_failures']}")
        logger.info(f"  Average processing time: {stats['avg_processing_time']:.4f}s")
        logger.info(
            f"  Throughput: {stats['processed_count'] / total_processing_time:.1f} items/sec"
        )

        # Performance assertions
        assert stats["processed_count"] > 800  # Should process most valid items
        assert stats["error_count"] > 0  # Should catch invalid data
        assert stats["avg_processing_time"] < 0.01  # Should be fast (< 10ms avg)
        assert stats["processed_count"] / total_processing_time > 100  # > 100 items/sec throughput

        # Data quality assertions
        assert len(processed_data) == stats["processed_count"]

        # Verify processed data quality
        for market_data in processed_data[:10]:  # Check first 10 items
            assert isinstance(market_data, MarketData)
            assert market_data.close > 0
            assert market_data.volume > 0
            assert market_data.symbol in symbols

        logger.info("Data processing pipeline throughput test completed")

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_data_pipeline_backpressure_handling(self):
        """Test data pipeline handling of backpressure and queue management."""

        class BackpressureDataProcessor:
            def __init__(self, max_queue_size: int = 100):
                self.input_queue = asyncio.Queue(maxsize=max_queue_size)
                self.processed_items = []
                self.dropped_items = 0
                self.processing_delay = 0.01  # 10ms processing delay
                self.running = False

            async def add_data(self, data: dict[str, Any]) -> bool:
                """Add data to processing queue."""
                try:
                    self.input_queue.put_nowait(data)
                    return True
                except asyncio.QueueFull:
                    self.dropped_items += 1
                    logger.warning("Queue full, dropping data item")
                    return False

            async def start_processing(self):
                """Start processing data from queue."""
                self.running = True

                while self.running:
                    try:
                        # Get data with timeout
                        data = await asyncio.wait_for(self.input_queue.get(), timeout=1.0)

                        # Simulate processing delay
                        await asyncio.sleep(self.processing_delay)

                        # Process data
                        processed_item = {
                            "symbol": data["symbol"],
                            "price": Decimal(str(data["price"])),
                            "processed_at": datetime.now(timezone.utc),
                        }

                        self.processed_items.append(processed_item)

                        # Mark task done
                        self.input_queue.task_done()

                    except asyncio.TimeoutError:
                        # No data available, continue
                        continue
                    except Exception as e:
                        logger.error(f"Processing error: {e}")
                        self.input_queue.task_done()

            def stop_processing(self):
                """Stop processing."""
                self.running = False

            def get_stats(self) -> dict[str, Any]:
                """Get processing statistics."""
                return {
                    "queue_size": self.input_queue.qsize(),
                    "processed_count": len(self.processed_items),
                    "dropped_count": self.dropped_items,
                    "processing_delay": self.processing_delay,
                }

        # Test with normal processing speed
        processor = BackpressureDataProcessor(max_queue_size=50)

        # Start processing
        processing_task = asyncio.create_task(processor.start_processing())

        # Generate data faster than processing speed
        data_generation_rate = 0.005  # 5ms between items (200/sec)
        processing_rate = 0.01  # 10ms per item (100/sec)

        # This should cause backpressure
        test_items = []
        for i in range(200):  # Generate 200 items
            data_item = {
                "symbol": f"TEST{i % 5}",  # 5 different symbols
                "price": 100 + i,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            added = await processor.add_data(data_item)
            test_items.append((data_item, added))

            await asyncio.sleep(data_generation_rate)

        # Wait for processing to complete
        await asyncio.sleep(2.0)  # Allow processing to catch up
        processor.stop_processing()

        # Wait for processor to stop
        processing_task.cancel()
        try:
            await processing_task
        except asyncio.CancelledError:
            pass

        # Analyze results
        stats = processor.get_stats()

        logger.info("Backpressure test results:")
        logger.info(f"  Total items generated: {len(test_items)}")
        logger.info(f"  Items processed: {stats['processed_count']}")
        logger.info(f"  Items dropped: {stats['dropped_count']}")
        logger.info(f"  Final queue size: {stats['queue_size']}")

        # Verify backpressure handling
        assert stats["dropped_count"] > 0  # Should drop some items due to backpressure
        assert stats["processed_count"] + stats["dropped_count"] + stats["queue_size"] == len(
            test_items
        )

        # Test with slower data generation (should not cause backpressure)
        processor2 = BackpressureDataProcessor(max_queue_size=50)
        processing_task2 = asyncio.create_task(processor2.start_processing())

        # Generate data slower than processing speed
        slower_rate = 0.02  # 20ms between items (50/sec)

        for i in range(50):
            data_item = {
                "symbol": f"SLOW{i % 3}",
                "price": 200 + i,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await processor2.add_data(data_item)
            await asyncio.sleep(slower_rate)

        # Wait for processing
        await asyncio.sleep(1.0)
        processor2.stop_processing()

        processing_task2.cancel()
        try:
            await processing_task2
        except asyncio.CancelledError:
            pass

        stats2 = processor2.get_stats()

        logger.info("Slower generation test results:")
        logger.info(f"  Items processed: {stats2['processed_count']}")
        logger.info(f"  Items dropped: {stats2['dropped_count']}")

        # Should not drop items with slower generation
        assert stats2["dropped_count"] == 0  # No backpressure
        assert stats2["processed_count"] == 50  # All items processed

        logger.info("Backpressure handling test completed")


class TestRealTimeRiskMonitoring:
    """Test real-time risk monitoring integration."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_real_time_portfolio_risk_updates(self):
        """Test real-time portfolio risk calculation updates."""

        class RealTimeRiskMonitor:
            def __init__(self):
                self.portfolio_value_history = []
                self.risk_metrics_history = []
                self.risk_alerts = []
                self.update_count = 0

            async def update_portfolio_value(self, new_value: Decimal):
                """Update portfolio value and recalculate risk metrics."""
                self.update_count += 1
                timestamp = datetime.now(timezone.utc)

                self.portfolio_value_history.append({"value": new_value, "timestamp": timestamp})

                # Calculate risk metrics
                risk_metrics = await self._calculate_risk_metrics()
                self.risk_metrics_history.append(risk_metrics)

                # Check for risk alerts
                await self._check_risk_alerts(risk_metrics)

            async def _calculate_risk_metrics(self) -> dict[str, Any]:
                """Calculate current risk metrics."""
                if len(self.portfolio_value_history) < 2:
                    return {
                        "var_1d": Decimal("0"),
                        "drawdown": Decimal("0"),
                        "volatility": Decimal("0"),
                        "timestamp": datetime.now(timezone.utc),
                    }

                # Calculate simple metrics
                values = [item["value"] for item in self.portfolio_value_history]
                current_value = values[-1]
                max_value = max(values)

                # Drawdown
                drawdown = (
                    (max_value - current_value) / max_value if max_value > 0 else Decimal("0")
                )

                # Simple volatility (using recent values)
                if len(values) >= 10:
                    recent_values = values[-10:]
                    returns = []
                    for i in range(1, len(recent_values)):
                        ret = (recent_values[i] - recent_values[i - 1]) / recent_values[i - 1]
                        returns.append(float(ret))

                    if returns:
                        import statistics

                        volatility = (
                            Decimal(str(statistics.stdev(returns)))
                            if len(returns) > 1
                            else Decimal("0")
                        )
                    else:
                        volatility = Decimal("0")
                else:
                    volatility = Decimal("0")

                # Simple VaR (2% of portfolio value)
                var_1d = current_value * Decimal("0.02")

                return {
                    "var_1d": var_1d,
                    "drawdown": drawdown,
                    "volatility": volatility,
                    "current_value": current_value,
                    "timestamp": datetime.now(timezone.utc),
                }

            async def _check_risk_alerts(self, risk_metrics: dict[str, Any]):
                """Check for risk alerts based on current metrics."""
                # Drawdown alert
                if risk_metrics["drawdown"] > Decimal("0.1"):  # 10% drawdown
                    alert = {
                        "type": "drawdown",
                        "level": "high" if risk_metrics["drawdown"] > Decimal("0.15") else "medium",
                        "value": risk_metrics["drawdown"],
                        "timestamp": risk_metrics["timestamp"],
                        "message": f"Portfolio drawdown: {risk_metrics['drawdown']:.1%}",
                    }
                    self.risk_alerts.append(alert)
                    logger.warning(alert["message"])

                # Volatility alert
                if risk_metrics["volatility"] > Decimal("0.05"):  # 5% volatility
                    alert = {
                        "type": "volatility",
                        "level": "high"
                        if risk_metrics["volatility"] > Decimal("0.1")
                        else "medium",
                        "value": risk_metrics["volatility"],
                        "timestamp": risk_metrics["timestamp"],
                        "message": f"High portfolio volatility: {risk_metrics['volatility']:.1%}",
                    }
                    self.risk_alerts.append(alert)
                    logger.warning(alert["message"])

            def get_current_risk_level(self) -> str:
                """Get current overall risk level."""
                if not self.risk_metrics_history:
                    return "unknown"

                latest_metrics = self.risk_metrics_history[-1]

                # High risk conditions
                if latest_metrics["drawdown"] > Decimal("0.15") or latest_metrics[
                    "volatility"
                ] > Decimal("0.1"):
                    return "high"

                # Medium risk conditions
                elif latest_metrics["drawdown"] > Decimal("0.05") or latest_metrics[
                    "volatility"
                ] > Decimal("0.03"):
                    return "medium"

                return "low"

        risk_monitor = RealTimeRiskMonitor()

        # Simulate real-time portfolio value updates
        base_value = Decimal("100000")  # $100k starting value

        # Scenario 1: Gradual decline (should trigger drawdown alert)
        portfolio_values = []
        for i in range(20):
            # Simulate gradual 15% decline
            decline_factor = 1 - (i * 0.008)  # 0.8% decline per step
            current_value = base_value * Decimal(str(decline_factor))
            portfolio_values.append(current_value)

            await risk_monitor.update_portfolio_value(current_value)
            await asyncio.sleep(0.01)  # Small delay between updates

        # Check that drawdown alert was triggered
        drawdown_alerts = [
            alert for alert in risk_monitor.risk_alerts if alert["type"] == "drawdown"
        ]
        assert len(drawdown_alerts) > 0, "Should have triggered drawdown alert"

        # Scenario 2: High volatility (rapid up/down movements)
        import random

        for i in range(20):
            # Add random volatility
            volatility_factor = 1 + (random.uniform(-0.1, 0.1))  # ±10% random moves
            current_value = portfolio_values[-1] * Decimal(str(volatility_factor))
            portfolio_values.append(current_value)

            await risk_monitor.update_portfolio_value(current_value)
            await asyncio.sleep(0.01)

        # Check overall risk level
        current_risk_level = risk_monitor.get_current_risk_level()

        logger.info("Real-time risk monitoring results:")
        logger.info(f"  Total updates: {risk_monitor.update_count}")
        logger.info(f"  Risk alerts triggered: {len(risk_monitor.risk_alerts)}")
        logger.info(f"  Current risk level: {current_risk_level}")

        # Verify monitoring functionality
        assert risk_monitor.update_count == 40  # 20 + 20 updates
        assert len(risk_monitor.risk_alerts) > 0  # Should have triggered alerts
        assert current_risk_level in ["low", "medium", "high"]

        # Check alert types
        alert_types = set(alert["type"] for alert in risk_monitor.risk_alerts)
        logger.info(f"  Alert types triggered: {alert_types}")

        # Should have at least drawdown alerts
        assert "drawdown" in alert_types

        logger.info("Real-time portfolio risk monitoring test completed")
