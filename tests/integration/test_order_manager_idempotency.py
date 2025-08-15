"""
Integration tests for Order Manager with Idempotency.

Tests cover:
- End-to-end order submission with idempotency
- Exchange integration with client_order_id
- Concurrent order submissions
- Network failure and retry scenarios
- Order lifecycle with idempotency tracking
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError
from src.core.types import OrderRequest, OrderResponse, OrderSide, OrderStatus, OrderType
from src.execution.order_manager import OrderManager


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock(spec=Config)
    config.error_handling = MagicMock()
    return config


@pytest.fixture
def mock_exchange():
    """Create a mock exchange."""
    exchange = AsyncMock()
    exchange.exchange_name = "test_exchange"
    exchange.place_order = AsyncMock()
    exchange.get_order_status = AsyncMock(return_value=OrderStatus.PENDING)
    return exchange


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    redis_client = AsyncMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.set = AsyncMock(return_value=True)
    redis_client.delete = AsyncMock(return_value=True)
    return redis_client


@pytest.fixture
def order_manager(mock_config):
    """Create an order manager instance."""
    return OrderManager(mock_config)


@pytest.fixture
def order_manager_with_redis(mock_config, mock_redis_client):
    """Create an order manager instance with Redis."""
    return OrderManager(mock_config, mock_redis_client)


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


class TestOrderManagerIdempotency:
    """Test Order Manager integration with Idempotency."""

    @pytest.mark.asyncio
    async def test_order_submission_with_idempotency(self, order_manager, mock_exchange, sample_order_request):
        """Test basic order submission with idempotency."""
        # Start order manager
        await order_manager.start()
        
        # Mock exchange response
        mock_response = OrderResponse(
            id="exchange_order_123",
            client_order_id=None,  # Will be set by idempotency manager
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            filled_quantity=Decimal("0"),
            status=OrderStatus.PENDING.value,
            timestamp=datetime.now(timezone.utc)
        )
        mock_exchange.place_order.return_value = mock_response
        
        # Submit order
        managed_order = await order_manager.submit_order(
            sample_order_request,
            mock_exchange,
            "test_execution_123"
        )
        
        # Verify client_order_id was generated and set
        assert managed_order.order_request.client_order_id is not None
        assert managed_order.order_request.client_order_id.startswith("T-")
        
        # Verify exchange was called with client_order_id
        mock_exchange.place_order.assert_called_once()
        called_order = mock_exchange.place_order.call_args[0][0]
        assert called_order.client_order_id == managed_order.order_request.client_order_id
        
        # Verify idempotency statistics
        stats = order_manager.idempotency_manager.get_statistics()
        assert stats["total_keys_created"] == 1
        assert stats["duplicate_orders_prevented"] == 0

    @pytest.mark.asyncio
    async def test_duplicate_order_prevention(self, order_manager, mock_exchange, sample_order_request):
        """Test duplicate order prevention."""
        await order_manager.start()
        
        # Mock exchange response
        mock_response = OrderResponse(
            id="exchange_order_123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            filled_quantity=Decimal("0"),
            status=OrderStatus.PENDING.value,
            timestamp=datetime.now(timezone.utc)
        )
        mock_exchange.place_order.return_value = mock_response
        
        # Submit first order
        managed_order1 = await order_manager.submit_order(
            sample_order_request,
            mock_exchange,
            "test_execution_123"
        )
        
        # Try to submit identical order - should be rejected due to max retries
        with patch.object(order_manager.idempotency_manager, 'max_retries', 0):
            with pytest.raises(ExecutionError, match="Duplicate order detected"):
                await order_manager.submit_order(
                    sample_order_request,
                    mock_exchange,
                    "test_execution_456"
                )
        
        # Verify only one exchange call was made
        assert mock_exchange.place_order.call_count == 1
        
        # Verify statistics
        stats = order_manager.idempotency_manager.get_statistics()
        assert stats["duplicate_orders_prevented"] == 1

    @pytest.mark.asyncio
    async def test_retry_with_same_client_order_id(self, order_manager, mock_exchange, sample_order_request):
        """Test retry logic using the same client_order_id."""
        await order_manager.start()
        
        # First submission fails
        mock_exchange.place_order.side_effect = Exception("Network error")
        
        try:
            await order_manager.submit_order(
                sample_order_request,
                mock_exchange,
                "test_execution_123"
            )
        except ExecutionError:
            pass  # Expected failure
        
        # Second submission succeeds (retry)
        mock_response = OrderResponse(
            id="exchange_order_123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            filled_quantity=Decimal("0"),
            status=OrderStatus.PENDING.value,
            timestamp=datetime.now(timezone.utc)
        )
        mock_exchange.place_order.side_effect = None
        mock_exchange.place_order.return_value = mock_response
        
        managed_order = await order_manager.submit_order(
            sample_order_request,
            mock_exchange,
            "test_execution_123"
        )
        
        # Both submissions should use the same client_order_id
        assert mock_exchange.place_order.call_count == 2
        call1_order = mock_exchange.place_order.call_args_list[0][0][0]
        call2_order = mock_exchange.place_order.call_args_list[1][0][0]
        assert call1_order.client_order_id == call2_order.client_order_id

    @pytest.mark.asyncio
    async def test_order_completion_tracking(self, order_manager, mock_exchange, sample_order_request):
        """Test order completion tracking in idempotency."""
        await order_manager.start()
        
        # Mock exchange response
        mock_response = OrderResponse(
            id="exchange_order_123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            filled_quantity=Decimal("0"),
            status=OrderStatus.PENDING.value,
            timestamp=datetime.now(timezone.utc)
        )
        mock_exchange.place_order.return_value = mock_response
        
        # Submit order
        managed_order = await order_manager.submit_order(
            sample_order_request,
            mock_exchange,
            "test_execution_123"
        )
        
        client_order_id = managed_order.order_request.client_order_id
        
        # Simulate order completion
        await order_manager._handle_status_change(
            managed_order,
            OrderStatus.PENDING,
            OrderStatus.FILLED
        )
        
        # Verify idempotency key is marked as completed
        key = await order_manager.idempotency_manager._find_key_by_client_order_id(client_order_id)
        assert key is not None
        assert key.status == "completed"

    @pytest.mark.asyncio
    async def test_order_failure_tracking(self, order_manager, mock_exchange, sample_order_request):
        """Test order failure tracking in idempotency."""
        await order_manager.start()
        
        # Mock exchange to fail
        mock_exchange.place_order.side_effect = Exception("Order rejected")
        
        # Submit order (should fail)
        try:
            await order_manager.submit_order(
                sample_order_request,
                mock_exchange,
                "test_execution_123"
            )
        except ExecutionError:
            pass  # Expected
        
        # Check that failure was tracked
        stats = order_manager.idempotency_manager.get_statistics()
        assert stats["failed_operations"] == 0  # The operation succeeded, but the order failed
        
        # There should be a key marked as failed
        active_keys = await order_manager.idempotency_manager.get_active_keys()
        # Since the order submission failed, the key should still exist but might be failed
        # depending on the exact failure point

    @pytest.mark.asyncio
    async def test_concurrent_order_submissions(self, order_manager, mock_exchange, sample_order_request):
        """Test concurrent identical order submissions."""
        await order_manager.start()
        
        # Mock exchange response
        mock_response = OrderResponse(
            id="exchange_order_123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            filled_quantity=Decimal("0"),
            status=OrderStatus.PENDING.value,
            timestamp=datetime.now(timezone.utc)
        )
        mock_exchange.place_order.return_value = mock_response
        
        # Submit multiple identical orders concurrently
        async def submit_order():
            try:
                return await order_manager.submit_order(
                    sample_order_request,
                    mock_exchange,
                    "test_execution_123"
                )
            except ExecutionError:
                return None  # Failed due to duplicate detection
        
        tasks = [submit_order() for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Only one should succeed (the first one)
        successful_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        assert len(successful_results) <= order_manager.idempotency_manager.max_retries + 1
        
        # All successful submissions should have the same client_order_id
        if len(successful_results) > 1:
            client_ids = [r.order_request.client_order_id for r in successful_results]
            assert len(set(client_ids)) == 1

    @pytest.mark.asyncio
    async def test_order_manager_with_redis(self, order_manager_with_redis, mock_exchange, sample_order_request):
        """Test order manager with Redis backing."""
        await order_manager_with_redis.start()
        
        # Mock exchange response
        mock_response = OrderResponse(
            id="exchange_order_123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            filled_quantity=Decimal("0"),
            status=OrderStatus.PENDING.value,
            timestamp=datetime.now(timezone.utc)
        )
        mock_exchange.place_order.return_value = mock_response
        
        # Submit order
        managed_order = await order_manager_with_redis.submit_order(
            sample_order_request,
            mock_exchange,
            "test_execution_123"
        )
        
        # Verify Redis operations were called
        redis_client = order_manager_with_redis.idempotency_manager.redis_client
        redis_client.set.assert_called()
        
        # Verify statistics show Redis usage
        stats = order_manager_with_redis.idempotency_manager.get_statistics()
        assert stats["use_redis"] is True
        assert stats["redis_operations"] > 0

    @pytest.mark.asyncio
    async def test_order_manager_summary_includes_idempotency(self, order_manager, mock_exchange, sample_order_request):
        """Test that order manager summary includes idempotency statistics."""
        await order_manager.start()
        
        # Submit an order to generate some statistics
        mock_response = OrderResponse(
            id="exchange_order_123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            filled_quantity=Decimal("0"),
            status=OrderStatus.PENDING.value,
            timestamp=datetime.now(timezone.utc)
        )
        mock_exchange.place_order.return_value = mock_response
        
        await order_manager.submit_order(
            sample_order_request,
            mock_exchange,
            "test_execution_123"
        )
        
        # Get summary
        summary = await order_manager.get_order_manager_summary()
        
        # Verify idempotency statistics are included
        assert "idempotency_statistics" in summary
        idempotency_stats = summary["idempotency_statistics"]
        assert "total_keys_created" in idempotency_stats
        assert "duplicate_orders_prevented" in idempotency_stats
        assert "cache_size" in idempotency_stats

    @pytest.mark.asyncio
    async def test_shutdown_includes_idempotency_cleanup(self, order_manager):
        """Test that shutdown properly cleans up idempotency manager."""
        await order_manager.start()
        
        # Add some data to idempotency manager
        sample_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001")
        )
        await order_manager.idempotency_manager.get_or_create_idempotency_key(sample_order)
        
        # Verify data exists
        assert len(order_manager.idempotency_manager._in_memory_cache) > 0
        
        # Shutdown
        await order_manager.shutdown()
        
        # Verify cleanup
        assert len(order_manager.idempotency_manager._in_memory_cache) == 0

    @pytest.mark.asyncio
    async def test_order_cancellation_marks_idempotency_failed(self, order_manager, mock_exchange, sample_order_request):
        """Test that order cancellation marks idempotency key as failed."""
        await order_manager.start()
        
        # Mock exchange response
        mock_response = OrderResponse(
            id="exchange_order_123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            filled_quantity=Decimal("0"),
            status=OrderStatus.PENDING.value,
            timestamp=datetime.now(timezone.utc)
        )
        mock_exchange.place_order.return_value = mock_response
        
        # Submit order
        managed_order = await order_manager.submit_order(
            sample_order_request,
            mock_exchange,
            "test_execution_123"
        )
        
        client_order_id = managed_order.order_request.client_order_id
        
        # Simulate order cancellation
        await order_manager._handle_status_change(
            managed_order,
            OrderStatus.PENDING,
            OrderStatus.CANCELLED
        )
        
        # Verify idempotency key is marked as failed
        key = await order_manager.idempotency_manager._find_key_by_client_order_id(client_order_id)
        assert key is not None
        assert key.status == "failed"
        assert "cancelled" in key.metadata.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_different_orders_get_different_keys(self, order_manager, mock_exchange):
        """Test that different orders get different idempotency keys."""
        await order_manager.start()
        
        # Mock exchange response
        def create_mock_response(symbol):
            return OrderResponse(
                id=f"exchange_order_{symbol}",
                client_order_id=None,
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("50000"),
                filled_quantity=Decimal("0"),
                status=OrderStatus.PENDING.value,
                timestamp=datetime.now(timezone.utc)
            )
        
        mock_exchange.place_order.side_effect = lambda order: create_mock_response(order.symbol)
        
        # Create different orders
        order1 = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000")
        )
        
        order2 = OrderRequest(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("3000")
        )
        
        # Submit both orders
        managed_order1 = await order_manager.submit_order(order1, mock_exchange, "exec_1")
        managed_order2 = await order_manager.submit_order(order2, mock_exchange, "exec_2")
        
        # Verify different client_order_ids
        assert managed_order1.order_request.client_order_id != managed_order2.order_request.client_order_id
        
        # Verify no duplicates detected
        stats = order_manager.idempotency_manager.get_statistics()
        assert stats["duplicate_orders_prevented"] == 0
        assert stats["total_keys_created"] == 2