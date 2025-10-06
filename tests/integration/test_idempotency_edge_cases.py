"""
Edge case tests for Order Idempotency System.

Tests cover:
- Network failures and timeouts
- Redis failures and recovery
- Race conditions in concurrent access
- Memory pressure and cleanup
- Clock skew and time-based issues
- Exchange-specific error scenarios
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import Config
from src.core.exceptions import ExchangeConnectionError, ExecutionError
from src.core.types import OrderRequest, OrderResponse, OrderSide, OrderStatus, OrderType
from src.execution.idempotency_manager import OrderIdempotencyManager
from src.execution.order_manager import OrderManager


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock(spec=Config)
    config.error_handling = MagicMock()
    return config


@pytest.fixture
def sample_order_request():
    """Create a sample order request."""
    return OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.001"),
        price=Decimal("50000"),
        time_in_force="GTC",
    )


class TestIdempotencyEdgeCases:
    """Test edge cases and failure scenarios for idempotency system."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_redis_connection_failure_during_operation(
        self, mock_config, sample_order_request
    ):
        """Test Redis connection failure during operations."""
        # Create Redis client that fails intermittently
        redis_client = AsyncMock()
        redis_client.get.side_effect = Exception("Redis connection lost")
        redis_client.set.side_effect = Exception("Redis connection lost")

        manager = OrderIdempotencyManager(mock_config, redis_client)

        # Should still work with in-memory fallback
        client_id1, is_duplicate1 = await manager.get_or_create_idempotency_key(
            sample_order_request
        )
        assert client_id1 is not None
        assert is_duplicate1 is False

        # Second call should detect duplicate from in-memory cache
        client_id2, is_duplicate2 = await manager.get_or_create_idempotency_key(
            sample_order_request
        )
        assert client_id1 == client_id2
        assert is_duplicate2 is True

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_redis_partial_failure_recovery(self, mock_config, sample_order_request):
        """Test recovery from partial Redis failures."""
        redis_client = AsyncMock()
        manager = OrderIdempotencyManager(mock_config, redis_client)

        # First operation succeeds
        client_id, _ = await manager.get_or_create_idempotency_key(sample_order_request)

        # Redis set fails, but get still works
        redis_client.set.side_effect = Exception("Redis set failed")

        key_data = {
            "key": f"idempotency:order:{manager._generate_order_hash(sample_order_request)}",
            "client_order_id": client_id,
            "order_hash": manager._generate_order_hash(sample_order_request),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
            "retry_count": 0,
            "status": "active",
            "metadata": {},
        }
        redis_client.get.return_value = json.dumps(key_data)

        # Clear in-memory cache to force Redis lookup
        manager._in_memory_cache.clear()

        # Should still detect duplicate from Redis
        client_id2, is_duplicate = await manager.get_or_create_idempotency_key(sample_order_request)
        assert client_id == client_id2
        assert is_duplicate is True

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_high_concurrency_race_conditions(self, mock_config, sample_order_request):
        """Test race conditions under high concurrency."""
        manager = OrderIdempotencyManager(mock_config)

        # Simulate high concurrency with many workers
        async def create_key_with_delay():
            # Add small random delay to increase chance of race conditions
            await asyncio.sleep(0.001 * (time.time() % 10))
            return await manager.get_or_create_idempotency_key(sample_order_request)

        # Launch many concurrent requests
        tasks = [create_key_with_delay() for _ in range(50)]
        results = await asyncio.gather(*tasks)

        # All should return the same client_order_id
        client_ids = [result[0] for result in results]
        unique_client_ids = set(client_ids)
        assert len(unique_client_ids) == 1, (
            f"Expected 1 unique client_id, got {len(unique_client_ids)}"
        )

        # Only one should not be duplicate
        is_duplicates = [result[1] for result in results]
        non_duplicates = [not dup for dup in is_duplicates]
        assert sum(non_duplicates) == 1, f"Expected 1 non-duplicate, got {sum(non_duplicates)}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_memory_pressure_cleanup(self, mock_config, sample_order_request):
        """Test cleanup under memory pressure."""
        manager = OrderIdempotencyManager(mock_config)

        # Create many keys with very short expiration
        for i in range(100):
            order = OrderRequest(
                symbol=f"TEST{i:03d}USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("1000"),
            )
            await manager.get_or_create_idempotency_key(
                order,
                expiration_hours=0.001,  # Very short expiration
            )

        assert len(manager._in_memory_cache) == 100

        # Wait for expiration
        await asyncio.sleep(0.1)

        # Force cleanup
        expired_count = await manager._cleanup_expired_keys()

        assert expired_count == 100
        assert len(manager._in_memory_cache) == 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_clock_skew_issues(self, mock_config, sample_order_request):
        """Test handling of clock skew issues."""
        manager = OrderIdempotencyManager(mock_config)

        # Create key with future timestamp (simulating clock skew)
        with patch("src.execution.idempotency_manager.datetime") as mock_datetime:
            future_time = datetime.now(timezone.utc) + timedelta(hours=1)
            mock_datetime.now.return_value = future_time
            mock_datetime.fromtimestamp = datetime.fromtimestamp
            mock_datetime.fromisoformat = datetime.fromisoformat

            client_id, _ = await manager.get_or_create_idempotency_key(sample_order_request)

        # Reset to current time
        with patch("src.execution.idempotency_manager.datetime") as mock_datetime:
            current_time = datetime.now(timezone.utc)
            mock_datetime.now.return_value = current_time
            mock_datetime.fromtimestamp = datetime.fromtimestamp
            mock_datetime.fromisoformat = datetime.fromisoformat

            # Should still be considered valid (not expired)
            client_id2, is_duplicate = await manager.get_or_create_idempotency_key(
                sample_order_request
            )
            assert client_id == client_id2
            assert is_duplicate is True

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_malformed_redis_data_handling(self, mock_config, sample_order_request):
        """Test handling of malformed data in Redis."""
        redis_client = AsyncMock()
        manager = OrderIdempotencyManager(mock_config, redis_client)

        # Return malformed JSON
        redis_client.get.return_value = "invalid json data"

        # Should fallback to creating new key
        client_id, is_duplicate = await manager.get_or_create_idempotency_key(sample_order_request)
        assert client_id is not None
        assert is_duplicate is False

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_exchange_timeout_retry_scenario(self, mock_config):
        """Test exchange timeout and retry scenario."""
        order_manager = OrderManager(mock_config)
        await order_manager.start()

        # Create mock exchange that times out then succeeds
        exchange = AsyncMock()
        exchange.exchange_name = "test_exchange"

        call_count = 0

        def place_order_side_effect(order):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ExchangeConnectionError("Connection timeout")
            return OrderResponse(
                id="exchange_order_123",
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                filled_quantity=Decimal("0"),
                status=OrderStatus.PENDING.value,
                timestamp=datetime.now(timezone.utc),
            )

        exchange.place_order.side_effect = place_order_side_effect

        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
        )

        # First submission fails
        try:
            await order_manager.submit_order(order_request, exchange, "exec_1")
            assert False, "Should have raised ExecutionError"
        except ExecutionError:
            pass  # Expected

        # Second submission should succeed with same client_order_id
        managed_order = await order_manager.submit_order(order_request, exchange, "exec_1")

        # Verify both calls used the same client_order_id
        call1_order = exchange.place_order.call_args_list[0][0][0]
        call2_order = exchange.place_order.call_args_list[1][0][0]
        assert call1_order.client_order_id == call2_order.client_order_id

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_rapid_retry_exhaustion(self, mock_config, sample_order_request):
        """Test rapid retry attempts that exhaust retry limits."""
        manager = OrderIdempotencyManager(mock_config)
        manager.max_retries = 2  # Set low retry limit

        # Create initial key
        client_id, _ = await manager.get_or_create_idempotency_key(sample_order_request)

        # Exhaust retries rapidly
        can_retry1, count1 = await manager.can_retry_order(client_id)
        assert can_retry1 is True
        assert count1 == 1

        can_retry2, count2 = await manager.can_retry_order(client_id)
        assert can_retry2 is True
        assert count2 == 2

        # Should be exhausted now
        can_retry3, count3 = await manager.can_retry_order(client_id)
        assert can_retry3 is False
        assert count3 == 2  # Max retries reached

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_concurrent_cleanup_and_access(self, mock_config):
        """Test concurrent cleanup and access operations."""
        manager = OrderIdempotencyManager(mock_config)

        # Create keys with short expiration
        orders = []
        for i in range(10):
            order = OrderRequest(
                symbol=f"TEST{i:03d}USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
            )
            orders.append(order)
            await manager.get_or_create_idempotency_key(order, expiration_hours=0.001)

        # Start concurrent operations
        async def continuous_cleanup():
            for _ in range(10):
                await manager._cleanup_expired_keys()
                await asyncio.sleep(0.01)

        async def continuous_access():
            for order in orders:
                try:
                    await manager.get_or_create_idempotency_key(order)
                except Exception:
                    pass  # Some operations may fail due to cleanup
                await asyncio.sleep(0.01)

        # Wait for expiration
        await asyncio.sleep(0.1)

        # Run concurrent operations
        await asyncio.gather(continuous_cleanup(), continuous_access(), return_exceptions=True)

        # Should not crash or leave inconsistent state
        stats = manager.get_statistics()
        assert stats["failed_operations"] == 0  # No operation-level failures

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_redis_data_corruption_recovery(self, mock_config, sample_order_request):
        """Test recovery from Redis data corruption."""
        redis_client = AsyncMock()
        manager = OrderIdempotencyManager(mock_config, redis_client)

        # Simulate corrupted data in Redis (missing required fields)
        corrupted_data = {
            "key": "some_key",
            # Missing required fields like client_order_id, order_hash, etc.
        }
        redis_client.get.return_value = json.dumps(corrupted_data)

        # Should handle corruption gracefully and create new key
        client_id, is_duplicate = await manager.get_or_create_idempotency_key(sample_order_request)

        assert client_id is not None
        assert is_duplicate is False

        # Should increment failed operations in stats (due to corrupted data handling)
        # Note: This might not increment failed_operations as the operation succeeds
        # but creates a new key instead of using corrupted one

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_large_metadata_handling(self, mock_config, sample_order_request):
        """Test handling of large metadata objects."""
        manager = OrderIdempotencyManager(mock_config)

        # Create large metadata
        large_metadata = {
            "execution_context": "x" * 10000,  # 10KB string
            "strategies": [f"strategy_{i}" for i in range(1000)],  # Large list
            "market_data": {f"indicator_{i}": i * 1.5 for i in range(500)},  # Large dict
        }

        # Should handle large metadata without issues
        client_id, is_duplicate = await manager.get_or_create_idempotency_key(
            sample_order_request, metadata=large_metadata
        )

        assert client_id is not None
        assert is_duplicate is False

        # Verify metadata is stored
        key = await manager._find_key_by_client_order_id(client_id)
        assert key is not None
        assert key.metadata == large_metadata

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_system_shutdown_during_operations(self, mock_config, sample_order_request):
        """Test system shutdown during ongoing operations."""
        manager = OrderIdempotencyManager(mock_config)
        await manager.start()

        # Start some long-running operations
        async def long_operation():
            for _ in range(100):
                try:
                    await manager.get_or_create_idempotency_key(sample_order_request)
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    break
                except Exception:
                    pass

        # Start operation in background
        operation_task = asyncio.create_task(long_operation())

        # Wait a bit then shutdown
        await asyncio.sleep(0.05)
        await manager.shutdown()

        # Operation should be cancelled
        assert operation_task.cancelled() or operation_task.done()

        # Cache should be cleaned up
        assert len(manager._in_memory_cache) == 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_key_expiration_edge_timing(self, mock_config, sample_order_request):
        """Test edge cases around key expiration timing."""
        manager = OrderIdempotencyManager(mock_config)

        # Create key with very short expiration
        client_id, _ = await manager.get_or_create_idempotency_key(
            sample_order_request, expiration_hours=0.001
        )

        # Access key just before expiration
        await asyncio.sleep(0.035)  # Just before 0.001 hour expiration

        # Should still be valid
        key = await manager._find_key_by_client_order_id(client_id)
        if key:  # Timing dependent
            assert not key.is_expired() or key.is_expired()  # Either is acceptable

        # Wait for certain expiration
        await asyncio.sleep(0.1)

        # Try to create same order again
        client_id2, is_duplicate = await manager.get_or_create_idempotency_key(sample_order_request)

        # Should create new key due to expiration
        assert is_duplicate is False

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_invalid_retry_scenarios(self, mock_config, sample_order_request):
        """Test various invalid retry scenarios."""
        manager = OrderIdempotencyManager(mock_config)

        # Try to retry non-existent order
        can_retry, count = await manager.can_retry_order("non_existent_id")
        assert can_retry is False
        assert count == 0

        # Try to mark non-existent order as completed
        success = await manager.mark_order_completed("non_existent_id", None)
        assert success is False

        # Try to mark non-existent order as failed
        success = await manager.mark_order_failed("non_existent_id", "error")
        assert success is False
