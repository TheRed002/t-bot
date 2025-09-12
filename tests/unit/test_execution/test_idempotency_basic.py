"""Optimized unit tests for OrderIdempotencyManager."""

import logging
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

# Disable logging for performance
logging.disable(logging.CRITICAL)

# Pre-defined constants for faster test data creation
TEST_DECIMALS = {
    "ONE": Decimal("1.0"),
    "PRICE_50K": Decimal("50000.0")
}

# Cache common configurations for better performance
COMMON_ATTRS = {
    "symbol": "BTC/USDT",
    "client_order_id": "test_order_001",
    "order_id": "order_123",
    "error_msg": "Test error",
    "ttl_seconds": 3600
}

from src.core.config import Config
from src.core.types import OrderRequest, OrderSide, OrderType
from src.execution.idempotency_manager import OrderIdempotencyManager


class TestOrderIdempotencyBasic:
    """Basic test cases for OrderIdempotencyManager."""

    @pytest.fixture(scope="session")
    def config(self):
        """Create test configuration with cached values."""
        config = MagicMock(spec=Config)
        config.execution = MagicMock()
        config.execution.get = MagicMock(return_value=COMMON_ATTRS["ttl_seconds"])  # 1 hour TTL
        # Add additional required attributes for performance
        config.database = MagicMock()
        config.monitoring = MagicMock()
        config.redis = MagicMock()
        config.error_handling = MagicMock()
        return config

    @pytest.fixture(scope="session")
    def idempotency_manager(self, config):
        """Create OrderIdempotencyManager instance."""
        return OrderIdempotencyManager(config)

    @pytest.fixture(scope="session")
    def sample_order_request(self):
        """Create sample order request with cached constants."""
        return OrderRequest(
            symbol=COMMON_ATTRS["symbol"],
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=TEST_DECIMALS["ONE"],
            price=TEST_DECIMALS["PRICE_50K"],
            client_order_id=COMMON_ATTRS["client_order_id"],
        )

    def test_initialization(self, idempotency_manager, config):
        """Test idempotency manager initialization."""
        assert idempotency_manager.config == config
        assert hasattr(idempotency_manager, "ttl_seconds")

    @pytest.mark.asyncio
    async def test_get_or_create_idempotency_key_new(
        self, idempotency_manager, sample_order_request
    ):
        """Test creating new idempotency key."""
        key = await idempotency_manager.get_or_create_idempotency_key(
            sample_order_request.client_order_id, sample_order_request
        )

        assert key is not None
        assert key.client_order_id == sample_order_request.client_order_id

    @pytest.mark.asyncio
    async def test_mark_order_completed(self, idempotency_manager, sample_order_request):
        """Test marking order as completed."""
        # First create a key
        await idempotency_manager.get_or_create_idempotency_key(
            sample_order_request.client_order_id, sample_order_request
        )

        # Mark as completed
        result = await idempotency_manager.mark_order_completed(
            sample_order_request.client_order_id, COMMON_ATTRS["order_id"]
        )

        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_mark_order_failed(self, idempotency_manager):
        """Test marking order as failed."""
        result = await idempotency_manager.mark_order_failed(
            COMMON_ATTRS["client_order_id"], COMMON_ATTRS["error_msg"]
        )
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_can_retry_order(self, idempotency_manager):
        """Test checking if order can be retried."""
        can_retry, retry_count = await idempotency_manager.can_retry_order(
            COMMON_ATTRS["client_order_id"]
        )
        assert isinstance(can_retry, bool)
        assert isinstance(retry_count, int)

    @pytest.mark.asyncio
    async def test_get_active_keys(self, idempotency_manager):
        """Test getting active keys."""
        keys = await idempotency_manager.get_active_keys()
        assert isinstance(keys, list)

    @pytest.mark.asyncio
    async def test_force_expire_key(self, idempotency_manager):
        """Test force expiring a key."""
        result = await idempotency_manager.force_expire_key(
            COMMON_ATTRS["client_order_id"]
        )
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_start_and_stop(self, idempotency_manager):
        """Test starting and stopping the manager."""
        await idempotency_manager.start()
        assert idempotency_manager.running

        await idempotency_manager.stop()
        assert not idempotency_manager.running

    def test_hash_order_data(self, idempotency_manager, sample_order_request):
        """Test order data hashing."""
        hash_value = idempotency_manager._hash_order_data(sample_order_request.model_dump())
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0

    def test_generate_key(self, idempotency_manager):
        """Test key generation."""
        client_order_id = COMMON_ATTRS["client_order_id"]
        order_hash = "test_hash"
        key = idempotency_manager._generate_key(client_order_id, order_hash)
        assert isinstance(key, str)
        assert client_order_id in key

    @pytest.mark.asyncio
    async def test_manager_lifecycle(self, idempotency_manager, sample_order_request):
        """Test full manager lifecycle."""
        # Start manager
        await idempotency_manager.start()

        # Create idempotency key
        key = await idempotency_manager.get_or_create_idempotency_key(
            sample_order_request.client_order_id, sample_order_request
        )
        assert key is not None

        # Mark as completed
        await idempotency_manager.mark_order_completed(
            sample_order_request.client_order_id, COMMON_ATTRS["order_id"]
        )

        # Stop manager
        await idempotency_manager.stop()

    def test_memory_store_initialization(self, idempotency_manager):
        """Test that memory store is properly initialized."""
        # Check that the manager has required attributes for memory storage
        assert hasattr(idempotency_manager, "_memory_store")
        assert hasattr(idempotency_manager, "_lock")
