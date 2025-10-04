"""
Idempotency Integration Tests with Real Services

Tests idempotency mechanisms using real database and cache for operation tracking,
ensuring exactly-once execution semantics.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.core.types import OrderSide, OrderType
from src.execution.idempotency_manager import OrderIdempotencyManager
from tests.integration.infrastructure.service_factory import RealServiceFactory

logger = logging.getLogger(__name__)


class RealServiceIdempotencyTest:
    """Idempotency tests using real services."""

    async def setup_test_services(self, clean_database):
        """Setup real services for idempotency testing."""
        service_factory = RealServiceFactory()

        # Initialize core services with the clean database
        await service_factory.initialize_core_services(clean_database)

        # Create dependency container with real services
        container = await service_factory.create_dependency_container()

        # Get services from container
        self.database_service = container.get("DatabaseService")
        self.cache_manager = container.get("CacheManager")

        # Create mock execution operations for idempotency testing
        # Since this is testing idempotency, not actual order execution
        self._order_counter = 0

        # Create idempotency manager with real persistence
        from src.core.config import get_config
        config = get_config()

        # Get Redis client from cache manager for idempotency storage
        redis_client = await self.cache_manager.get_client() if hasattr(self.cache_manager, "get_client") else None

        self.idempotency_manager = OrderIdempotencyManager(
            config=config,
            redis_client=redis_client
        )

        # Store factory for cleanup
        self.service_factory = service_factory
        return container

    async def mock_submit_order(self, **order_data):
        """Mock order submission for idempotency testing."""
        self._order_counter += 1
        order_id = f"order_{self._order_counter}_{uuid.uuid4().hex[:8]}"

        # Simulate order processing time
        await asyncio.sleep(0.01)

        return {
            "order_id": order_id,
            "symbol": order_data.get("symbol"),
            "status": "submitted",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @pytest.mark.asyncio
    async def test_real_idempotent_order_submission(self, clean_database):
        """Test idempotent order submission with real persistence."""
        container = await self.setup_test_services(clean_database)

        try:
            idempotency_key = f"order_{uuid.uuid4()}"
            order_data = {
                "symbol": "BTC/USDT",
                "side": OrderSide.BUY,
                "order_type": OrderType.LIMIT,
                "quantity": Decimal("0.1"),
                "price": Decimal("45000.0")
            }

            # Test idempotency with simplified order data approach
            order_dict = {
                "symbol": "BTC/USDT",
                "side": "buy",
                "order_type": "limit",
                "quantity": "0.1",
                "price": "45000.0",
                "client_order_id": idempotency_key
            }

            # First submission - should create new entry (returns None for new orders)
            result1 = await self.idempotency_manager.check_and_store_order(
                client_order_id=idempotency_key,
                order_data=order_dict
            )
            assert result1 is None  # New orders return None

            # Second submission with same client_order_id - should detect duplicate
            result2 = await self.idempotency_manager.check_and_store_order(
                client_order_id=idempotency_key,
                order_data=order_dict
            )
            assert result2 == order_dict  # Duplicates return original order data

            # Test that the idempotency manager is using real persistence
            # Check if we can retrieve the order status
            order_status = await self.idempotency_manager.get_order_status(idempotency_key)
            assert order_status is not None

            # Verify idempotency is working by checking active keys
            active_keys = await self.idempotency_manager.get_active_keys()
            assert len(active_keys) > 0
            assert any(key["client_order_id"] == idempotency_key for key in active_keys)

            # Test basic infrastructure functionality
            assert self.idempotency_manager is not None
            logger.info("✅ Idempotency manager successfully created and tested with Redis persistence")

            logger.info("✅ Real idempotent order submission test passed")

        finally:
            # Cleanup using service factory
            await self.service_factory.cleanup()
            container.clear()

    @pytest.mark.asyncio
    async def test_real_concurrent_idempotent_operations(self, clean_database):
        """Test concurrent idempotent operations with real locking."""
        container = await self.setup_test_services(clean_database)

        try:
            base_client_order_id = f"concurrent_{uuid.uuid4()}"

            # Create same order data for all concurrent requests
            order_data = {
                "symbol": "ETH/USDT",
                "side": "buy",
                "order_type": "limit",
                "quantity": "1.0",
                "price": "3000.0"
            }

            # Launch multiple concurrent requests with DIFFERENT order data (different quantities)
            tasks = [
                self.idempotency_manager.check_and_store_order(
                    client_order_id=f"{base_client_order_id}_{i}",
                    order_data={
                        "symbol": "ETH/USDT",
                        "side": "buy",
                        "order_type": "limit",
                        "quantity": f"{1.0 + i * 0.1}",  # Different quantities
                        "price": "3000.0"
                    }
                )
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks)

            # All should be treated as new orders (return None) since they have different content
            assert all(r is None for r in results)

            # Now test with same client_order_id and same order content - should detect duplicates
            duplicate_client_order_id = f"duplicate_{uuid.uuid4()}"
            duplicate_order_data = {
                "symbol": "BTC/USDT",
                "side": "sell",
                "order_type": "market",
                "quantity": "0.5",
                "price": "50000.0"
            }

            same_id_tasks = [
                self.idempotency_manager.check_and_store_order(
                    client_order_id=duplicate_client_order_id,
                    order_data=duplicate_order_data
                )
                for _ in range(3)
            ]

            same_id_results = await asyncio.gather(*same_id_tasks)

            # First should be None (new), subsequent should return order_data (duplicates)
            assert same_id_results[0] is None  # New order
            assert same_id_results[1] == duplicate_order_data  # Duplicate
            assert same_id_results[2] == duplicate_order_data  # Duplicate

            logger.info("✅ Real concurrent idempotent operations test passed")

        finally:
            # Cleanup using service factory
            await self.service_factory.cleanup()
            container.clear()

    @pytest.mark.asyncio
    async def test_real_idempotency_expiration(self, clean_database):
        """Test idempotency key expiration with real TTL."""
        container = await self.setup_test_services(clean_database)

        try:
            client_order_id = f"expiring_{uuid.uuid4()}"

            # Create test order data
            order_data = {
                "symbol": "LTC/USDT",
                "side": "sell",
                "order_type": "market",
                "quantity": "2.0",
                "price": "100.0"
            }

            # First submission with short expiration (convert hours to appropriate value)
            result1 = await self.idempotency_manager.check_and_store_order(
                client_order_id=client_order_id,
                order_data=order_data,
                expiration_hours=1  # 1 hour TTL
            )
            assert result1 is None  # New order

            # Immediate retry should detect duplicate
            result2 = await self.idempotency_manager.check_and_store_order(
                client_order_id=client_order_id,
                order_data=order_data,
                expiration_hours=1
            )
            assert result2 == order_data  # Duplicate detected

            # Check order status
            status = await self.idempotency_manager.get_order_status(client_order_id)
            assert status is not None
            assert status["client_order_id"] == client_order_id
            assert status["status"] == "pending"

            # Test manual expiration (force expire the key)
            expired = await self.idempotency_manager.force_expire_key(client_order_id)
            assert expired is True

            # After expiration, should be treated as new order again
            result3 = await self.idempotency_manager.check_and_store_order(
                client_order_id=client_order_id,
                order_data=order_data,
                expiration_hours=1
            )
            assert result3 is None  # New order again after expiration

            logger.info("✅ Real idempotency expiration test passed")

        finally:
            # Cleanup using service factory
            await self.service_factory.cleanup()
            container.clear()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_comprehensive_real_idempotency(clean_database):
    """Run comprehensive idempotency tests with real services."""
    test = RealServiceIdempotencyTest()

    test_methods = [
        test.test_real_idempotent_order_submission,
        test.test_real_concurrent_idempotent_operations,
        test.test_real_idempotency_expiration,
    ]

    for test_method in test_methods:
        await test_method(clean_database)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
