"""
Unit tests for the Order Idempotency Manager.

Tests cover:
- Client order ID generation and uniqueness
- Duplicate order detection
- Retry logic with consistent client_order_ids
- Key expiration and cleanup
- Thread safety and async safety
- Redis fallback and in-memory caching
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError
from src.core.types import OrderRequest, OrderResponse, OrderSide, OrderStatus, OrderType
from src.execution.idempotency_manager import OrderIdempotencyManager


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock(spec=Config)
    config.error_handling = MagicMock()
    return config


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    redis_client = AsyncMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.set = AsyncMock(return_value=True)
    redis_client.delete = AsyncMock(return_value=True)
    return redis_client


@pytest.fixture
def sample_order_request():
    """Create a sample order request."""
    return OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.001"),
        price=Decimal("50000"),
        time_in_force="GTC"
    )


@pytest.fixture
def idempotency_manager(mock_config):
    """Create an idempotency manager instance."""
    return OrderIdempotencyManager(mock_config)


@pytest.fixture
def idempotency_manager_with_redis(mock_config, mock_redis_client):
    """Create an idempotency manager instance with Redis."""
    return OrderIdempotencyManager(mock_config, mock_redis_client)


class TestOrderIdempotencyManager:
    """Test the Order Idempotency Manager."""

    def test_initialization(self, mock_config):
        """Test idempotency manager initialization."""
        manager = OrderIdempotencyManager(mock_config)
        
        assert manager.config == mock_config
        assert manager.redis_client is None
        assert manager.use_redis is False
        assert manager.default_expiration_hours == 24
        assert manager.max_retries == 3
        assert len(manager._in_memory_cache) == 0

    def test_initialization_with_redis(self, mock_config, mock_redis_client):
        """Test idempotency manager initialization with Redis."""
        manager = OrderIdempotencyManager(mock_config, mock_redis_client)
        
        assert manager.redis_client == mock_redis_client
        assert manager.use_redis is True

    def test_generate_order_hash(self, idempotency_manager, sample_order_request):
        """Test order hash generation."""
        hash1 = idempotency_manager._generate_order_hash(sample_order_request)
        
        # Hash should be consistent for same order
        hash2 = idempotency_manager._generate_order_hash(sample_order_request)
        assert hash1 == hash2
        
        # Hash should be different for different orders
        different_order = OrderRequest(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1")
        )
        hash3 = idempotency_manager._generate_order_hash(different_order)
        assert hash1 != hash3

    def test_generate_client_order_id(self, idempotency_manager, sample_order_request):
        """Test client order ID generation."""
        client_id1 = idempotency_manager._generate_client_order_id(sample_order_request)
        client_id2 = idempotency_manager._generate_client_order_id(sample_order_request)
        
        # IDs should be unique
        assert client_id1 != client_id2
        
        # Should contain symbol and side prefixes
        assert "BTCUSD" in client_id1  # First 6 chars of symbol
        assert "-B-" in client_id1     # Buy side
        assert client_id1.startswith("T-")

    def test_generate_idempotency_key(self, idempotency_manager):
        """Test idempotency key generation."""
        order_hash = "test_hash_123"
        key = idempotency_manager._generate_idempotency_key(order_hash)
        
        assert key == f"idempotency:order:{order_hash}"

    @pytest.mark.asyncio
    async def test_get_or_create_new_key(self, idempotency_manager, sample_order_request):
        """Test creating a new idempotency key."""
        client_id, is_duplicate = await idempotency_manager.get_or_create_idempotency_key(
            sample_order_request
        )
        
        assert client_id is not None
        assert is_duplicate is False
        assert idempotency_manager.stats["total_keys_created"] == 1
        assert idempotency_manager.stats["cache_misses"] == 1

    @pytest.mark.asyncio
    async def test_get_or_create_duplicate_key(self, idempotency_manager, sample_order_request):
        """Test duplicate order detection."""
        # Create first key
        client_id1, is_duplicate1 = await idempotency_manager.get_or_create_idempotency_key(
            sample_order_request
        )
        
        # Try to create same order again
        client_id2, is_duplicate2 = await idempotency_manager.get_or_create_idempotency_key(
            sample_order_request
        )
        
        assert client_id1 == client_id2
        assert is_duplicate1 is False
        assert is_duplicate2 is True
        assert idempotency_manager.stats["duplicate_orders_prevented"] == 1

    @pytest.mark.asyncio
    async def test_invalid_order_validation(self, idempotency_manager):
        """Test validation of invalid orders."""
        # Order without symbol
        invalid_order = OrderRequest(
            symbol="",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001")
        )
        
        with pytest.raises(ExecutionError, match="Order must have symbol and quantity"):
            await idempotency_manager.get_or_create_idempotency_key(invalid_order)
        
        # Order with negative quantity
        invalid_order2 = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("-0.001")
        )
        
        with pytest.raises(ExecutionError, match="Order quantity must be positive"):
            await idempotency_manager.get_or_create_idempotency_key(invalid_order2)

    @pytest.mark.asyncio
    async def test_mark_order_completed(self, idempotency_manager, sample_order_request):
        """Test marking an order as completed."""
        # Create key
        client_id, _ = await idempotency_manager.get_or_create_idempotency_key(sample_order_request)
        
        # Create mock order response
        order_response = OrderResponse(
            id="test_order_123",
            client_order_id=client_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            filled_quantity=Decimal("0.001"),
            status=OrderStatus.FILLED.value,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Mark as completed
        success = await idempotency_manager.mark_order_completed(client_id, order_response)
        
        assert success is True
        
        # Verify key is marked as completed
        key = await idempotency_manager._find_key_by_client_order_id(client_id)
        assert key is not None
        assert key.status == "completed"

    @pytest.mark.asyncio
    async def test_mark_order_failed(self, idempotency_manager, sample_order_request):
        """Test marking an order as failed."""
        # Create key
        client_id, _ = await idempotency_manager.get_or_create_idempotency_key(sample_order_request)
        
        # Mark as failed
        error_message = "Network timeout"
        success = await idempotency_manager.mark_order_failed(client_id, error_message)
        
        assert success is True
        
        # Verify key is marked as failed
        key = await idempotency_manager._find_key_by_client_order_id(client_id)
        assert key is not None
        assert key.status == "failed"
        assert key.metadata["error"] == error_message

    @pytest.mark.asyncio
    async def test_retry_logic(self, idempotency_manager, sample_order_request):
        """Test retry logic and retry count limits."""
        # Create key
        client_id, _ = await idempotency_manager.get_or_create_idempotency_key(sample_order_request)
        
        # Test retries within limit
        for i in range(idempotency_manager.max_retries):
            can_retry, retry_count = await idempotency_manager.can_retry_order(client_id)
            assert can_retry is True
            assert retry_count == i + 1
        
        # Test retry beyond limit
        can_retry, retry_count = await idempotency_manager.can_retry_order(client_id)
        assert can_retry is False
        assert retry_count == idempotency_manager.max_retries

    @pytest.mark.asyncio
    async def test_key_expiration(self, idempotency_manager, sample_order_request):
        """Test key expiration logic."""
        # Create key with very short expiration (1 millisecond)
        client_id, _ = await idempotency_manager.get_or_create_idempotency_key(
            sample_order_request,
            expiration_hours=1/3600000  # 1 millisecond 
        )
        
        # Wait longer to ensure expiration
        await asyncio.sleep(0.01)
        
        # Try to create same order again - should create new key due to expiration
        client_id2, is_duplicate = await idempotency_manager.get_or_create_idempotency_key(
            sample_order_request
        )
        
        # Should not be considered duplicate due to expiration
        assert is_duplicate is False
        # New client ID should be generated
        assert client_id != client_id2

    @pytest.mark.asyncio
    async def test_cleanup_expired_keys(self, idempotency_manager, sample_order_request):
        """Test cleanup of expired keys."""
        # Create key with very short expiration
        await idempotency_manager.get_or_create_idempotency_key(
            sample_order_request,
            expiration_hours=1/3600000  # 1 millisecond
        )
        
        assert len(idempotency_manager._in_memory_cache) == 1
        
        # Wait for expiration
        await asyncio.sleep(0.01)
        
        # Run cleanup
        expired_count = await idempotency_manager._cleanup_expired_keys()
        
        assert expired_count == 1
        assert len(idempotency_manager._in_memory_cache) == 0

    @pytest.mark.asyncio
    async def test_redis_fallback(self, idempotency_manager_with_redis, sample_order_request):
        """Test Redis operations and fallback to in-memory cache."""
        manager = idempotency_manager_with_redis
        
        # Create key
        client_id, _ = await manager.get_or_create_idempotency_key(sample_order_request)
        
        # Verify Redis was called
        manager.redis_client.set.assert_called_once()
        
        # Clear in-memory cache to test Redis retrieval
        manager._in_memory_cache.clear()
        
        # Mock Redis return
        key_data = {
            "key": f"idempotency:order:{manager._generate_order_hash(sample_order_request)}",
            "client_order_id": client_id,
            "order_hash": manager._generate_order_hash(sample_order_request),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
            "retry_count": 0,
            "status": "active",
            "metadata": {}
        }
        manager.redis_client.get.return_value = json.dumps(key_data)
        
        # Try to create same order again
        client_id2, is_duplicate = await manager.get_or_create_idempotency_key(sample_order_request)
        
        assert client_id == client_id2
        assert is_duplicate is True
        manager.redis_client.get.assert_called()

    @pytest.mark.asyncio
    async def test_concurrent_access(self, idempotency_manager, sample_order_request):
        """Test thread safety with concurrent access."""
        async def create_key():
            return await idempotency_manager.get_or_create_idempotency_key(sample_order_request)
        
        # Run multiple concurrent requests
        tasks = [create_key() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should return the same client_order_id
        client_ids = [result[0] for result in results]
        is_duplicates = [result[1] for result in results]
        
        # First one should not be duplicate, others should be
        assert is_duplicates[0] is False
        assert all(is_dup is True for is_dup in is_duplicates[1:])
        
        # All client IDs should be the same
        assert len(set(client_ids)) == 1

    @pytest.mark.asyncio
    async def test_statistics(self, idempotency_manager, sample_order_request):
        """Test statistics collection."""
        # Create some keys and operations
        await idempotency_manager.get_or_create_idempotency_key(sample_order_request)
        await idempotency_manager.get_or_create_idempotency_key(sample_order_request)  # Duplicate
        
        stats = idempotency_manager.get_statistics()
        
        assert stats["total_keys_created"] == 1
        assert stats["duplicate_orders_prevented"] == 1
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["cache_size"] == 1
        assert stats["use_redis"] is False

    @pytest.mark.asyncio
    async def test_get_active_keys(self, idempotency_manager, sample_order_request):
        """Test getting active keys."""
        # Create key
        await idempotency_manager.get_or_create_idempotency_key(sample_order_request)
        
        active_keys = await idempotency_manager.get_active_keys()
        
        assert len(active_keys) == 1
        assert "client_order_id" in active_keys[0]
        assert "order_hash" in active_keys[0]
        assert "created_at" in active_keys[0]
        assert "status" in active_keys[0]

    @pytest.mark.asyncio
    async def test_force_expire_key(self, idempotency_manager, sample_order_request):
        """Test force expiring a key."""
        # Create key
        client_id, _ = await idempotency_manager.get_or_create_idempotency_key(sample_order_request)
        
        # Force expire
        success = await idempotency_manager.force_expire_key(client_id)
        assert success is True
        
        # Key should be gone
        key = await idempotency_manager._find_key_by_client_order_id(client_id)
        assert key is None

    @pytest.mark.asyncio
    async def test_start_and_shutdown(self, idempotency_manager):
        """Test start and shutdown lifecycle."""
        # Start manager
        await idempotency_manager.start()
        assert idempotency_manager._cleanup_started is True
        assert idempotency_manager._cleanup_task is not None
        
        # Shutdown manager
        await idempotency_manager.shutdown()
        assert len(idempotency_manager._in_memory_cache) == 0

    @pytest.mark.asyncio
    async def test_error_handling_in_operations(self, idempotency_manager):
        """Test error handling in various operations."""
        # Test with invalid client_order_id
        success = await idempotency_manager.mark_order_completed("invalid_id", None)
        assert success is False
        
        success = await idempotency_manager.mark_order_failed("invalid_id", "error")
        assert success is False
        
        can_retry, count = await idempotency_manager.can_retry_order("invalid_id")
        assert can_retry is False
        assert count == 0

    @pytest.mark.asyncio
    async def test_redis_error_handling(self, idempotency_manager_with_redis, sample_order_request):
        """Test Redis error handling and fallback."""
        manager = idempotency_manager_with_redis
        
        # Mock Redis to raise exception
        manager.redis_client.set.side_effect = Exception("Redis connection failed")
        
        # Should still work with in-memory cache
        client_id, is_duplicate = await manager.get_or_create_idempotency_key(sample_order_request)
        
        assert client_id is not None
        assert is_duplicate is False
        assert len(manager._in_memory_cache) == 1

    @pytest.mark.asyncio
    async def test_metadata_handling(self, idempotency_manager, sample_order_request):
        """Test metadata storage and retrieval."""
        metadata = {
            "execution_id": "test_execution_123",
            "exchange": "binance",
            "created_by": "test_system"
        }
        
        client_id, _ = await idempotency_manager.get_or_create_idempotency_key(
            sample_order_request,
            metadata=metadata
        )
        
        key = await idempotency_manager._find_key_by_client_order_id(client_id)
        assert key is not None
        assert key.metadata == metadata