"""Simple unit tests for IdempotencyManager."""

import logging
from unittest.mock import MagicMock

import pytest

# Disable logging for performance
logging.disable(logging.CRITICAL)

# Pre-defined constants for faster test data creation
TEST_DATA = {
    "CLIENT_ORDER_ID": "test_order_001",
    "ORDER_ID": "order_123",
    "SYMBOL": "BTCUSDT",
    "QUANTITY": "1.0",
    "STATUS_COMPLETED": "completed",
    "STATUS_FAILED": "failed",
    "TTL_SECONDS": 3600,
    "EMPTY_DICT": {},
    "TRUE_BOOL": True,
    "FALSE_BOOL": False
}

from src.core.config import Config
from src.execution.idempotency_manager import OrderIdempotencyManager


class TestOrderIdempotencyManager:
    """Test cases for OrderIdempotencyManager."""

    @pytest.fixture(scope="session")
    def config(self):
        """Create test configuration with pre-defined values."""
        config = MagicMock(spec=Config)
        config.execution = MagicMock()
        config.execution.get = MagicMock(return_value=TEST_DATA["TTL_SECONDS"])  # 1 hour TTL
        return config

    @pytest.fixture(scope="session")
    def idempotency_manager(self, config):
        """Create OrderIdempotencyManager instance."""
        return OrderIdempotencyManager(config)

    def test_initialization(self, idempotency_manager, config):
        """Test idempotency manager initialization."""
        assert idempotency_manager.config == config
        assert not idempotency_manager.use_redis  # Default is False
        assert isinstance(idempotency_manager._in_memory_cache, dict)

    def test_initialization_with_redis(self, config):
        """Test initialization with Redis."""
        redis_client = MagicMock()
        manager = OrderIdempotencyManager(config, redis_client=redis_client)
        assert manager.use_redis

    @pytest.mark.asyncio
    async def test_check_and_store_order_new(self, idempotency_manager):
        """Test checking and storing new order."""
        order_data = {"symbol": TEST_DATA["SYMBOL"], "quantity": TEST_DATA["QUANTITY"]}

        # Should return None for new order
        result = await idempotency_manager.check_and_store_order(TEST_DATA["CLIENT_ORDER_ID"], order_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_check_and_store_order_duplicate(self, idempotency_manager):
        """Test checking duplicate order."""
        client_order_id = "test_order_002"  # Use different ID to avoid conflicts
        order_data = {"symbol": TEST_DATA["SYMBOL"], "quantity": TEST_DATA["QUANTITY"]}

        # First call stores the order
        await idempotency_manager.check_and_store_order(client_order_id, order_data)

        # Second call should return the stored data
        result = await idempotency_manager.check_and_store_order(client_order_id, order_data)
        assert result is not None
        assert result["symbol"] == TEST_DATA["SYMBOL"]

    @pytest.mark.asyncio
    async def test_mark_order_completed(self, idempotency_manager):
        """Test marking order as completed."""
        client_order_id = "test_order_003"  # Use different ID to avoid conflicts
        order_data = {"symbol": TEST_DATA["SYMBOL"], "quantity": "2.0"}  # Different quantity to avoid duplicate hash

        # Store order first
        await idempotency_manager.check_and_store_order(client_order_id, order_data)

        # Mark as completed
        await idempotency_manager.mark_order_completed(client_order_id, TEST_DATA["ORDER_ID"])

        # Verify it's marked as completed
        result = await idempotency_manager.get_order_status(client_order_id)
        assert result["status"] == TEST_DATA["STATUS_COMPLETED"]

    @pytest.mark.asyncio
    async def test_mark_order_failed(self, idempotency_manager):
        """Test marking order as failed."""
        client_order_id = "test_order_004"  # Use different ID to avoid conflicts

        await idempotency_manager.mark_order_failed(client_order_id, "Insufficient funds")

        result = await idempotency_manager.get_order_status(client_order_id)
        assert result["status"] == TEST_DATA["STATUS_FAILED"]

    @pytest.mark.asyncio
    async def test_get_order_status_not_found(self, idempotency_manager):
        """Test getting status of non-existent order."""
        result = await idempotency_manager.get_order_status("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_cleanup_expired_orders(self, idempotency_manager):
        """Test cleanup of expired orders."""
        # This test ensures the method exists and can be called
        await idempotency_manager.cleanup_expired_orders()
        # No assertion needed - just test it doesn't crash

    def test_generate_idempotency_key(self, idempotency_manager):
        """Test idempotency key generation."""
        key = idempotency_manager._generate_idempotency_key(TEST_DATA["CLIENT_ORDER_ID"])
        assert key.startswith("idempotency:")
        assert TEST_DATA["CLIENT_ORDER_ID"] in key

    def test_memory_store_operations(self, idempotency_manager):
        """Test basic memory store operations."""
        key = "test_key"
        value = {"test": "data"}

        # Store value
        idempotency_manager.memory_store[key] = value

        # Retrieve value
        assert idempotency_manager.memory_store[key] == value

        # Check existence
        assert key in idempotency_manager.memory_store

    def test_ttl_configuration(self, idempotency_manager, config):
        """Test TTL configuration."""
        # TTL should be set from config
        config.execution.get.assert_called()
        assert hasattr(idempotency_manager, "ttl_seconds")

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, idempotency_manager):
        """Test concurrent order operations."""
        client_order_id = "test_order_concurrent"
        order_data = {"symbol": TEST_DATA["SYMBOL"], "quantity": "5.0"}  # Different quantity to avoid duplicate hash

        # Store order
        await idempotency_manager.check_and_store_order(client_order_id, order_data)

        # Check status while pending
        status = await idempotency_manager.get_order_status(client_order_id)
        assert status["status"] == "pending"

        # Mark completed
        await idempotency_manager.mark_order_completed(client_order_id, TEST_DATA["ORDER_ID"])

        # Verify final status
        final_status = await idempotency_manager.get_order_status(client_order_id)
        assert final_status["status"] == TEST_DATA["STATUS_COMPLETED"]
