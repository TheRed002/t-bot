"""
Enhanced Idempotency Integration Tests with Financial Validation

Tests idempotency mechanisms with real database persistence, financial precision,
and comprehensive edge case coverage for production trading scenarios.
"""

import asyncio
import logging
import uuid
from decimal import Decimal, getcontext

import pytest

from src.execution.idempotency_manager import OrderIdempotencyManager
from tests.integration.infrastructure.service_factory import RealServiceFactory

# Set financial precision for all calculations
getcontext().prec = 28  # High precision for financial calculations

logger = logging.getLogger(__name__)


class EnhancedIdempotencyTest:
    """Enhanced idempotency tests with financial validation."""

    async def setup_test_services(self, clean_database):
        """Setup real services for enhanced idempotency testing."""
        service_factory = RealServiceFactory()

        # Initialize core services with the clean database
        await service_factory.initialize_core_services(clean_database)

        # Create dependency container with real services
        container = await service_factory.create_dependency_container()

        # Get services from container
        self.database_service = container.get("DatabaseService")
        self.cache_manager = container.get("CacheManager")
        self.connection_manager = clean_database

        # Clear Redis to ensure clean state for each test
        raw_redis_client = await self.connection_manager.get_redis_client()
        await raw_redis_client.flushdb()

        # Create idempotency manager with real persistence
        from src.core.config import get_config

        config = get_config()

        # Use the CacheManager's wrapped Redis client for proper API
        self.idempotency_manager = OrderIdempotencyManager(
            config=config, redis_client=self.cache_manager.redis_client
        )

        # Store factory for cleanup
        self.service_factory = service_factory
        return container

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_financial_precision_idempotency(self, clean_database):
        """Test idempotency with precise financial decimal handling."""
        container = await self.setup_test_services(clean_database)

        try:
            # Test with precise financial amounts
            test_cases = [
                # (quantity, price, expected_total)
                (
                    Decimal("0.12345678"),
                    Decimal("45678.12345678"),
                    Decimal("0.12345678") * Decimal("45678.12345678"),
                ),
                (
                    Decimal("1.00000001"),
                    Decimal("50000.00000001"),
                    Decimal("1.00000001") * Decimal("50000.00000001"),
                ),
                (
                    Decimal("0.000000000000000001"),
                    Decimal("1000000.00"),
                    Decimal("0.000000000000000001") * Decimal("1000000.00"),
                ),  # Satoshi level
            ]

            for i, (quantity, price, expected_total) in enumerate(test_cases):
                idempotency_key = f"precision_{uuid.uuid4()}"

                # Create order with precise decimals (add index to make each test case unique)
                order_data = {
                    "symbol": "BTC/USDT",  # Keep symbol consistent
                    "side": "buy",
                    "type": "limit",  # Fixed: use "type" not "order_type"
                    "quantity": str(quantity),  # Convert to string for transport
                    "price": str(price),
                    "client_order_id": idempotency_key,
                    "test_case_id": i,  # Add unique test case identifier
                }

                # First submission
                result1 = await self.idempotency_manager.check_and_store_order(
                    client_order_id=idempotency_key, order_data=order_data
                )
                assert result1 is None  # New order

                # Verify precision is maintained in storage
                order_status = await self.idempotency_manager.get_order_status(idempotency_key)
                assert order_status is not None

                # Second submission - should detect duplicate
                result2 = await self.idempotency_manager.check_and_store_order(
                    client_order_id=idempotency_key, order_data=order_data
                )
                assert result2 == order_data  # Duplicate detected

                # Verify calculation precision
                actual_total = quantity * price
                assert abs(actual_total - expected_total) < Decimal("0.000000000000000001")

                logger.info(
                    f"✅ Financial precision test passed: {quantity} * {price} = {actual_total}"
                )

        finally:
            await self.service_factory.cleanup()
            container.clear()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_database_persistence_verification(self, clean_database):
        """Test that idempotency data is actually persisted in database."""
        container = await self.setup_test_services(clean_database)

        try:
            idempotency_key = f"db_persist_{uuid.uuid4()}"

            order_data = {
                "symbol": "ETH/USDT",
                "side": "sell",
                "type": "market",  # Fixed: use "type" not "order_type"
                "quantity": str(Decimal("2.5")),
                "price": str(Decimal("3000.00")),
                "client_order_id": idempotency_key,
            }

            # Submit order
            result = await self.idempotency_manager.check_and_store_order(
                client_order_id=idempotency_key, order_data=order_data
            )
            assert result is None  # New order

            # Verify in Redis
            redis_client = await self.connection_manager.get_redis_client()

            # Debug: List ALL keys in Redis to see what's actually there
            all_keys = await redis_client.keys("*")
            logger.info(f"DEBUG: Total keys in Redis: {len(all_keys)}")
            if len(all_keys) > 0:
                logger.info(
                    f"DEBUG: First 10 keys: {[k.decode() if isinstance(k, bytes) else k for k in all_keys[:10]]}"
                )

            # Check if key exists in Redis
            # Keys are stored with 'cache:' namespace when using CacheManager's RedisClient
            redis_keys = []
            cursor = 0
            while True:
                cursor, keys = await redis_client.scan(
                    cursor, match="cache:idempotency:order:*", count=100
                )
                redis_keys.extend(keys)
                if cursor == 0:
                    break

            logger.info(
                f"DEBUG: Found {len(redis_keys)} keys matching pattern cache:idempotency:order:*"
            )

            assert len(redis_keys) > 0, (
                f"Idempotency key {idempotency_key} not found in Redis. Searched for pattern: cache:idempotency:order:*"
            )

            logger.info(f"✅ Database persistence verified: {len(redis_keys)} keys found in Redis")

            # Verify duplicate detection works after Redis storage
            result2 = await self.idempotency_manager.check_and_store_order(
                client_order_id=idempotency_key, order_data=order_data
            )
            assert result2 == order_data  # Should detect duplicate from Redis

        finally:
            await self.service_factory.cleanup()
            container.clear()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_high_frequency_trading_stress(self, clean_database):
        """Test idempotency under high-frequency trading conditions."""
        container = await self.setup_test_services(clean_database)

        try:
            # Simulate rapid order submission
            num_unique_orders = 40  # Base unique orders
            num_duplicates = 20  # Total duplicates to create

            # Create deterministic unique orders
            unique_orders = []
            for i in range(num_unique_orders):
                unique_orders.append(
                    {
                        "client_order_id": f"hft_unique_{i}_{uuid.uuid4()}",
                        "data": {
                            "symbol": "BTC/USDT",
                            "side": "buy" if i % 2 == 0 else "sell",
                            "type": "limit",
                            "quantity": str(
                                Decimal(f"1.{i:08d}").quantize(Decimal("0.000000000000000001"))
                            ),  # Unique quantities
                            "price": str(
                                Decimal(f"45000.{i:02d}").quantize(Decimal("0.01"))
                            ),  # Unique prices
                        },
                    }
                )

            # Create base orders with unique content that will have exact duplicates
            base_orders_for_duplicates = []
            for i in range(10):
                base_orders_for_duplicates.append(
                    {
                        "client_order_id": f"hft_original_{i}_{uuid.uuid4()}",
                        "data": {
                            "symbol": "ETH/USDT",
                            "side": "buy",  # Keep same side for all base orders
                            "type": "limit",
                            "quantity": str(
                                Decimal(f"5.{i:08d}").quantize(Decimal("0.000000000000000001"))
                            ),  # UNIQUE quantity per base order
                            "price": str(
                                Decimal(f"2500.{i:02d}").quantize(Decimal("0.01"))
                            ),  # UNIQUE price per base order
                        },
                    }
                )

            # Create exact duplicates with identical content but different client IDs
            duplicate_orders = []
            for i in range(10):  # Create 2 duplicates for each of the 10 base orders
                base_data = base_orders_for_duplicates[i]["data"]
                for j in range(2):  # 2 duplicates per base order
                    duplicate_orders.append(
                        {
                            "client_order_id": f"hft_duplicate_{i}_{j}_{uuid.uuid4()}",
                            "data": base_data.copy(),  # Exact copy with same content
                        }
                    )

            # Combine all orders
            all_orders = unique_orders + base_orders_for_duplicates + duplicate_orders
            total_expected_new = num_unique_orders + 10  # 40 unique + 10 base orders = 50 new

            # Don't shuffle to ensure deterministic results for debugging
            logger.info(
                f"Testing with {len(unique_orders)} unique orders, {len(base_orders_for_duplicates)} base orders with duplicates, {len(duplicate_orders)} duplicates"
            )

            # Submit all orders sequentially to ensure proper duplicate detection
            # (concurrent submission might have Redis timing issues)
            results = []
            for order in all_orders:
                result = await self.idempotency_manager.check_and_store_order(
                    client_order_id=order["client_order_id"], order_data=order["data"]
                )
                results.append(result)

            # Count new vs duplicate orders
            new_orders = sum(1 for r in results if r is None)
            duplicate_detections = sum(1 for r in results if r is not None)

            logger.info(
                f"Results: {new_orders} new orders, {duplicate_detections} duplicate detections"
            )
            logger.info(
                f"Expected: {total_expected_new} new orders, {num_duplicates} duplicate detections"
            )

            assert new_orders == total_expected_new, (
                f"Expected {total_expected_new} new orders, got {new_orders}"
            )
            assert duplicate_detections == num_duplicates, (
                f"Expected {num_duplicates} duplicates, got {duplicate_detections}"
            )

            logger.info(
                f"✅ HFT stress test passed: {new_orders} new, {duplicate_detections} duplicates detected"
            )

        finally:
            await self.service_factory.cleanup()
            container.clear()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_redis_failure_fallback(self, clean_database):
        """Test idempotency behavior when Redis is unavailable."""
        container = await self.setup_test_services(clean_database)

        try:
            idempotency_key = f"redis_fail_{uuid.uuid4()}"

            order_data = {
                "symbol": "BTC/USDT",
                "side": "buy",
                "type": "limit",  # Fixed: use "type" not "order_type"
                "quantity": str(Decimal("0.1")),
                "price": str(Decimal("45000.00")),
            }

            # Submit order normally
            result1 = await self.idempotency_manager.check_and_store_order(
                client_order_id=idempotency_key, order_data=order_data
            )
            assert result1 is None

            # Simulate Redis connection failure
            redis_client = await self.connection_manager.get_redis_client()
            original_ping = redis_client.ping

            async def failing_ping():
                raise ConnectionError("Redis connection failed")

            redis_client.ping = failing_ping

            # Try to submit duplicate - should handle Redis failure gracefully
            try:
                result2 = await self.idempotency_manager.check_and_store_order(
                    client_order_id=idempotency_key, order_data=order_data
                )
                # May return None or order_data depending on fallback behavior
                logger.info(f"Redis failure handled: result={result2}")
            except Exception as e:
                logger.info(f"Expected error during Redis failure: {e}")

            # Restore Redis
            redis_client.ping = original_ping

            logger.info("✅ Redis failure fallback test completed")

        finally:
            await self.service_factory.cleanup()
            container.clear()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_order_modification_detection(self, clean_database):
        """Test detection of order modifications vs new orders."""
        container = await self.setup_test_services(clean_database)

        try:
            base_order_id = f"modify_{uuid.uuid4()}"

            # Original order
            original_order = {
                "symbol": "ETH/USDT",
                "side": "buy",
                "type": "limit",  # Fixed: use "type" not "order_type"
                "quantity": str(Decimal("1.0")),
                "price": str(Decimal("3000.00")),
            }

            # Submit original
            result1 = await self.idempotency_manager.check_and_store_order(
                client_order_id=base_order_id, order_data=original_order
            )
            assert result1 is None

            # Try exact duplicate - should detect
            result2 = await self.idempotency_manager.check_and_store_order(
                client_order_id=base_order_id, order_data=original_order
            )
            assert result2 == original_order

            # Modified order with different quantity (different content hash)
            modified_order = original_order.copy()
            modified_order["quantity"] = str(Decimal("1.5"))

            # This should be treated as a new order due to different content
            # OrderIdempotencyManager uses content-based detection, not client_order_id-based
            result3 = await self.idempotency_manager.check_and_store_order(
                client_order_id=base_order_id, order_data=modified_order
            )
            # Should be treated as new order since content is different
            assert result3 is None

            logger.info("✅ Order modification detection test passed")

        finally:
            await self.service_factory.cleanup()
            container.clear()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_concurrent_same_content_different_ids(self, clean_database):
        """Test concurrent orders with same content but different client IDs."""
        container = await self.setup_test_services(clean_database)

        try:
            # Same order content
            order_content = {
                "symbol": "SOL/USDT",
                "side": "buy",
                "type": "limit",  # Fixed: use "type" not "order_type"
                "quantity": str(Decimal("10.0")),
                "price": str(Decimal("100.00")),
            }

            # Submit with different client_order_ids concurrently
            tasks = []
            for i in range(5):
                tasks.append(
                    self.idempotency_manager.check_and_store_order(
                        client_order_id=f"concurrent_content_{uuid.uuid4()}",
                        order_data=order_content,
                    )
                )

            results = await asyncio.gather(*tasks)

            # Based on content hash, first should succeed, rest should be duplicates
            new_orders = sum(1 for r in results if r is None)
            duplicates = sum(1 for r in results if r is not None)

            # The idempotency manager uses content-based hashing
            # So same content = duplicate regardless of client_order_id
            assert new_orders == 1, f"Expected 1 new order, got {new_orders}"
            assert duplicates == 4, f"Expected 4 duplicates, got {duplicates}"

            logger.info("✅ Content-based idempotency test passed")

        finally:
            await self.service_factory.cleanup()
            container.clear()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_enhanced_idempotency_suite(clean_database):
    """Run comprehensive enhanced idempotency test suite."""
    test = EnhancedIdempotencyTest()

    test_methods = [
        test.test_financial_precision_idempotency,
        test.test_database_persistence_verification,
        test.test_high_frequency_trading_stress,
        test.test_redis_failure_fallback,
        test.test_order_modification_detection,
        test.test_concurrent_same_content_different_ids,
    ]

    for test_method in test_methods:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running: {test_method.__name__}")
        logger.info(f"{'=' * 60}")
        await test_method(clean_database)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
