"""
Comprehensive unit tests for OrderManager.

Tests critical order lifecycle management, validation, and state tracking
for high-frequency trading scenarios.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
from typing import Dict, List

from src.execution.order_manager import OrderManager
from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError, OrderRejectionError
from src.core.types import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    OrderRequest,
    OrderResponse,
    Position,
)


@pytest.fixture
def mock_config():
    """Mock configuration for OrderManager tests."""
    config = Mock(spec=Config)
    config.execution = Mock()
    config.execution.max_open_orders = 100
    config.execution.order_timeout = 300
    config.execution.max_order_size = Decimal('1000000')
    config.execution.min_order_size = Decimal('10')
    config.risk = Mock()
    config.risk.max_position_size = Decimal('10000')
    return config


@pytest.fixture
def order_manager(mock_config):
    """Create OrderManager instance for testing."""
    return OrderManager(mock_config)


@pytest.fixture
def sample_order_request():
    """Sample order request for testing."""
    return OrderRequest(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("50000.0"),
        time_in_force=TimeInForce.GTC,
        client_order_id="test_order_001"
    )


@pytest.fixture
def sample_order():
    """Sample order for testing."""
    return Order(
        id="order_123",
        client_order_id="test_order_001",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("50000.0"),
        status=OrderStatus.PENDING,
        time_in_force=TimeInForce.GTC,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


class TestOrderManagerInitialization:
    """Test OrderManager initialization and setup."""

    def test_initialization_with_valid_config(self, mock_config):
        """Test successful initialization with valid configuration."""
        manager = OrderManager(mock_config)
        
        assert manager.config == mock_config
        assert manager.orders == {}
        assert manager.positions == {}
        assert manager.order_sequence == 0

    def test_initialization_missing_config(self):
        """Test initialization fails without configuration."""
        with pytest.raises(TypeError):
            OrderManager(None)

    def test_configuration_validation(self, mock_config):
        """Test that configuration parameters are validated."""
        # Test invalid max_open_orders
        mock_config.execution.max_open_orders = -1
        
        with pytest.raises(ValidationError):
            OrderManager(mock_config)


class TestOrderCreation:
    """Test order creation and validation."""

    @pytest.mark.asyncio
    async def test_create_valid_order(self, order_manager, sample_order_request):
        """Test creation of a valid order."""
        order = await order_manager.create_order(sample_order_request)
        
        assert order.symbol == sample_order_request.symbol
        assert order.side == sample_order_request.side
        assert order.quantity == sample_order_request.quantity
        assert order.price == sample_order_request.price
        assert order.status == OrderStatus.PENDING
        assert order.id is not None
        assert order.created_at is not None

    @pytest.mark.asyncio
    async def test_create_order_generates_unique_ids(self, order_manager, sample_order_request):
        """Test that each order gets a unique ID."""
        order1 = await order_manager.create_order(sample_order_request)
        order2 = await order_manager.create_order(sample_order_request)
        
        assert order1.id != order2.id
        assert order1.client_order_id != order2.client_order_id

    @pytest.mark.asyncio
    async def test_create_order_validation_zero_quantity(self, order_manager, sample_order_request):
        """Test validation fails for zero quantity."""
        sample_order_request.quantity = Decimal("0")
        
        with pytest.raises(ValidationError, match="quantity"):
            await order_manager.create_order(sample_order_request)

    @pytest.mark.asyncio
    async def test_create_order_validation_negative_price(self, order_manager, sample_order_request):
        """Test validation fails for negative price."""
        sample_order_request.price = Decimal("-100")
        
        with pytest.raises(ValidationError, match="price"):
            await order_manager.create_order(sample_order_request)

    @pytest.mark.asyncio
    async def test_create_order_validation_oversized(self, order_manager, sample_order_request):
        """Test validation fails for oversized orders."""
        sample_order_request.quantity = Decimal("2000000")  # Exceeds max_order_size
        
        with pytest.raises(ValidationError, match="size"):
            await order_manager.create_order(sample_order_request)

    @pytest.mark.asyncio
    async def test_create_order_validation_undersized(self, order_manager, sample_order_request):
        """Test validation fails for undersized orders."""
        sample_order_request.quantity = Decimal("5")  # Below min_order_size
        
        with pytest.raises(ValidationError, match="size"):
            await order_manager.create_order(sample_order_request)


class TestOrderLifecycle:
    """Test order lifecycle management."""

    @pytest.mark.asyncio
    async def test_submit_order(self, order_manager, sample_order):
        """Test order submission."""
        # Add order to manager
        order_manager.orders[sample_order.id] = sample_order
        
        with patch.object(order_manager, '_send_to_exchange', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = OrderResponse(
                order_id=sample_order.id,
                status=OrderStatus.NEW,
                timestamp=datetime.now(timezone.utc)
            )
            
            response = await order_manager.submit_order(sample_order.id)
            
            assert response.order_id == sample_order.id
            assert response.status == OrderStatus.NEW
            mock_send.assert_called_once_with(sample_order)

    @pytest.mark.asyncio
    async def test_cancel_order(self, order_manager, sample_order):
        """Test order cancellation."""
        # Setup order as active
        sample_order.status = OrderStatus.NEW
        order_manager.orders[sample_order.id] = sample_order
        
        with patch.object(order_manager, '_send_cancel_to_exchange', new_callable=AsyncMock) as mock_cancel:
            mock_cancel.return_value = True
            
            result = await order_manager.cancel_order(sample_order.id)
            
            assert result is True
            assert sample_order.status == OrderStatus.CANCELLED
            mock_cancel.assert_called_once_with(sample_order.id)

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, order_manager):
        """Test cancellation of non-existent order fails."""
        with pytest.raises(ExecutionError, match="not found"):
            await order_manager.cancel_order("nonexistent_id")

    @pytest.mark.asyncio
    async def test_cancel_already_filled_order(self, order_manager, sample_order):
        """Test cancellation of already filled order fails."""
        sample_order.status = OrderStatus.FILLED
        order_manager.orders[sample_order.id] = sample_order
        
        with pytest.raises(ExecutionError, match="cannot be cancelled"):
            await order_manager.cancel_order(sample_order.id)

    @pytest.mark.asyncio
    async def test_order_fill_update(self, order_manager, sample_order):
        """Test order fill status updates."""
        order_manager.orders[sample_order.id] = sample_order
        
        fill_data = {
            "order_id": sample_order.id,
            "filled_quantity": Decimal("0.5"),
            "remaining_quantity": Decimal("0.5"),
            "average_price": Decimal("50000.0"),
            "status": OrderStatus.PARTIALLY_FILLED
        }
        
        await order_manager.update_order_fill(fill_data)
        
        assert sample_order.filled_quantity == Decimal("0.5")
        assert sample_order.remaining_quantity == Decimal("0.5")
        assert sample_order.status == OrderStatus.PARTIALLY_FILLED

    @pytest.mark.asyncio
    async def test_order_complete_fill(self, order_manager, sample_order):
        """Test complete order fill."""
        order_manager.orders[sample_order.id] = sample_order
        
        fill_data = {
            "order_id": sample_order.id,
            "filled_quantity": Decimal("1.0"),
            "remaining_quantity": Decimal("0.0"),
            "average_price": Decimal("50000.0"),
            "status": OrderStatus.FILLED
        }
        
        await order_manager.update_order_fill(fill_data)
        
        assert sample_order.filled_quantity == Decimal("1.0")
        assert sample_order.remaining_quantity == Decimal("0.0")
        assert sample_order.status == OrderStatus.FILLED


class TestOrderTracking:
    """Test order tracking and state management."""

    @pytest.mark.asyncio
    async def test_get_order_by_id(self, order_manager, sample_order):
        """Test retrieval of order by ID."""
        order_manager.orders[sample_order.id] = sample_order
        
        retrieved_order = order_manager.get_order(sample_order.id)
        
        assert retrieved_order == sample_order
        assert retrieved_order.id == sample_order.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_order(self, order_manager):
        """Test retrieval of non-existent order returns None."""
        retrieved_order = order_manager.get_order("nonexistent_id")
        assert retrieved_order is None

    @pytest.mark.asyncio
    async def test_get_orders_by_symbol(self, order_manager):
        """Test retrieval of orders by symbol."""
        # Create orders for different symbols
        btc_order = Order(
            id="btc_1", symbol="BTC/USDT", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=Decimal("1.0"),
            price=Decimal("50000.0"), status=OrderStatus.NEW
        )
        eth_order = Order(
            id="eth_1", symbol="ETH/USDT", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=Decimal("10.0"),
            price=Decimal("3000.0"), status=OrderStatus.NEW
        )
        
        order_manager.orders[btc_order.id] = btc_order
        order_manager.orders[eth_order.id] = eth_order
        
        btc_orders = order_manager.get_orders_by_symbol("BTC/USDT")
        eth_orders = order_manager.get_orders_by_symbol("ETH/USDT")
        
        assert len(btc_orders) == 1
        assert len(eth_orders) == 1
        assert btc_orders[0].symbol == "BTC/USDT"
        assert eth_orders[0].symbol == "ETH/USDT"

    @pytest.mark.asyncio
    async def test_get_orders_by_status(self, order_manager):
        """Test retrieval of orders by status."""
        # Create orders with different statuses
        new_order = Order(
            id="new_1", symbol="BTC/USDT", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=Decimal("1.0"),
            price=Decimal("50000.0"), status=OrderStatus.NEW
        )
        filled_order = Order(
            id="filled_1", symbol="BTC/USDT", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=Decimal("1.0"),
            price=Decimal("50000.0"), status=OrderStatus.FILLED
        )
        
        order_manager.orders[new_order.id] = new_order
        order_manager.orders[filled_order.id] = filled_order
        
        new_orders = order_manager.get_orders_by_status(OrderStatus.NEW)
        filled_orders = order_manager.get_orders_by_status(OrderStatus.FILLED)
        
        assert len(new_orders) == 1
        assert len(filled_orders) == 1
        assert new_orders[0].status == OrderStatus.NEW
        assert filled_orders[0].status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_get_active_orders(self, order_manager):
        """Test retrieval of active orders only."""
        # Create orders with different statuses
        active_statuses = [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING]
        inactive_statuses = [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
        
        for i, status in enumerate(active_statuses + inactive_statuses):
            order = Order(
                id=f"order_{i}", symbol="BTC/USDT", side=OrderSide.BUY,
                order_type=OrderType.LIMIT, quantity=Decimal("1.0"),
                price=Decimal("50000.0"), status=status
            )
            order_manager.orders[order.id] = order
        
        active_orders = order_manager.get_active_orders()
        
        assert len(active_orders) == 3  # Only active statuses
        for order in active_orders:
            assert order.status in active_statuses


class TestPositionManagement:
    """Test position tracking and management."""

    @pytest.mark.asyncio
    async def test_update_position_on_fill(self, order_manager, sample_order):
        """Test position update when order is filled."""
        # Setup initial position
        order_manager.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=Decimal("0.5"),
            average_price=Decimal("49000.0"),
            unrealized_pnl=Decimal("0.0")
        )
        
        # Simulate order fill
        sample_order.status = OrderStatus.FILLED
        sample_order.filled_quantity = Decimal("1.0")
        sample_order.average_price = Decimal("50000.0")
        
        await order_manager._update_position_on_fill(sample_order)
        
        position = order_manager.positions["BTC/USDT"]
        assert position.quantity == Decimal("1.5")  # 0.5 + 1.0
        # Average price should be updated based on weighted average

    @pytest.mark.asyncio
    async def test_create_new_position_on_fill(self, order_manager, sample_order):
        """Test creation of new position when none exists."""
        # Simulate order fill with no existing position
        sample_order.status = OrderStatus.FILLED
        sample_order.filled_quantity = Decimal("1.0")
        sample_order.average_price = Decimal("50000.0")
        
        await order_manager._update_position_on_fill(sample_order)
        
        position = order_manager.positions["BTC/USDT"]
        assert position.quantity == Decimal("1.0")
        assert position.average_price == Decimal("50000.0")
        assert position.side == OrderSide.LONG  # BUY order creates LONG position

    @pytest.mark.asyncio
    async def test_close_position_on_opposite_fill(self, order_manager):
        """Test position closure with opposite side order."""
        # Setup existing long position
        order_manager.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=Decimal("1.0"),
            average_price=Decimal("50000.0")
        )
        
        # Create sell order to close position
        sell_order = Order(
            id="sell_1", symbol="BTC/USDT", side=OrderSide.SELL,
            order_type=OrderType.MARKET, quantity=Decimal("1.0"),
            status=OrderStatus.FILLED, filled_quantity=Decimal("1.0"),
            average_price=Decimal("51000.0")
        )
        
        await order_manager._update_position_on_fill(sell_order)
        
        # Position should be closed (quantity = 0)
        position = order_manager.positions["BTC/USDT"]
        assert position.quantity == Decimal("0.0")

    @pytest.mark.asyncio
    async def test_get_position(self, order_manager):
        """Test position retrieval."""
        test_position = Position(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            quantity=Decimal("1.0"),
            average_price=Decimal("50000.0")
        )
        order_manager.positions["BTC/USDT"] = test_position
        
        retrieved_position = order_manager.get_position("BTC/USDT")
        
        assert retrieved_position == test_position
        assert retrieved_position.symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_get_all_positions(self, order_manager):
        """Test retrieval of all positions."""
        positions_data = [
            ("BTC/USDT", Decimal("1.0"), Decimal("50000.0")),
            ("ETH/USDT", Decimal("10.0"), Decimal("3000.0")),
            ("ADA/USDT", Decimal("1000.0"), Decimal("1.0"))
        ]
        
        for symbol, qty, price in positions_data:
            order_manager.positions[symbol] = Position(
                symbol=symbol,
                side=OrderSide.LONG,
                quantity=qty,
                average_price=price
            )
        
        all_positions = order_manager.get_all_positions()
        
        assert len(all_positions) == 3
        symbols = [pos.symbol for pos in all_positions]
        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols
        assert "ADA/USDT" in symbols


class TestRiskControls:
    """Test risk control mechanisms."""

    @pytest.mark.asyncio
    async def test_order_size_limit_enforcement(self, order_manager, sample_order_request):
        """Test that order size limits are enforced."""
        # Try to create order exceeding position size limit
        sample_order_request.quantity = Decimal("15000")  # Exceeds max_position_size
        
        with pytest.raises(ValidationError, match="position size"):
            await order_manager.create_order(sample_order_request)

    @pytest.mark.asyncio
    async def test_max_open_orders_limit(self, order_manager, mock_config):
        """Test maximum open orders limit."""
        mock_config.execution.max_open_orders = 2
        order_manager.config = mock_config
        
        # Create maximum allowed orders
        for i in range(2):
            order = Order(
                id=f"order_{i}", symbol="BTC/USDT", side=OrderSide.BUY,
                order_type=OrderType.LIMIT, quantity=Decimal("1.0"),
                price=Decimal("50000.0"), status=OrderStatus.NEW
            )
            order_manager.orders[order.id] = order
        
        # Try to create one more order
        request = OrderRequest(
            symbol="BTC/USDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"), price=Decimal("50000.0")
        )
        
        with pytest.raises(ExecutionError, match="maximum.*open orders"):
            await order_manager.create_order(request)

    @pytest.mark.asyncio
    async def test_duplicate_client_order_id_prevention(self, order_manager, sample_order_request):
        """Test prevention of duplicate client order IDs."""
        # Create first order
        await order_manager.create_order(sample_order_request)
        
        # Try to create second order with same client_order_id
        with pytest.raises(ValidationError, match="client.*order.*id.*exists"):
            await order_manager.create_order(sample_order_request)


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_exchange_rejection_handling(self, order_manager, sample_order):
        """Test handling of exchange order rejections."""
        order_manager.orders[sample_order.id] = sample_order
        
        with patch.object(order_manager, '_send_to_exchange', new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = OrderRejectionError(
                "Order rejected by exchange", rejection_reason="Insufficient funds"
            )
            
            with pytest.raises(OrderRejectionError):
                await order_manager.submit_order(sample_order.id)
            
            # Verify order status is updated
            assert sample_order.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_network_error_handling(self, order_manager, sample_order):
        """Test handling of network errors during order submission."""
        order_manager.orders[sample_order.id] = sample_order
        
        with patch.object(order_manager, '_send_to_exchange', new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = ExecutionError(
                "Network timeout", error_code="NET_001", retryable=True
            )
            
            with pytest.raises(ExecutionError):
                await order_manager.submit_order(sample_order.id)

    @pytest.mark.asyncio
    async def test_invalid_order_status_transition(self, order_manager, sample_order):
        """Test handling of invalid order status transitions."""
        # Set order to FILLED status
        sample_order.status = OrderStatus.FILLED
        order_manager.orders[sample_order.id] = sample_order
        
        # Try to update to PARTIALLY_FILLED (invalid transition)
        fill_data = {
            "order_id": sample_order.id,
            "status": OrderStatus.PARTIALLY_FILLED
        }
        
        with pytest.raises(ExecutionError, match="invalid.*status.*transition"):
            await order_manager.update_order_fill(fill_data)


class TestConcurrentOperations:
    """Test concurrent operations and thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_order_creation(self, order_manager):
        """Test concurrent order creation."""
        import asyncio
        
        async def create_order(i):
            request = OrderRequest(
                symbol="BTC/USDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"), price=Decimal("50000.0"),
                client_order_id=f"concurrent_{i}"
            )
            return await order_manager.create_order(request)
        
        # Create multiple orders concurrently
        tasks = [create_order(i) for i in range(5)]
        orders = await asyncio.gather(*tasks)
        
        # Verify all orders were created with unique IDs
        order_ids = [order.id for order in orders]
        assert len(set(order_ids)) == 5  # All IDs are unique
        
        # Verify all orders are tracked
        assert len(order_manager.orders) == 5

    @pytest.mark.asyncio
    async def test_concurrent_order_cancellation(self, order_manager):
        """Test concurrent order cancellations."""
        import asyncio
        
        # Setup orders
        orders = []
        for i in range(3):
            order = Order(
                id=f"cancel_test_{i}", symbol="BTC/USDT", side=OrderSide.BUY,
                order_type=OrderType.LIMIT, quantity=Decimal("1.0"),
                price=Decimal("50000.0"), status=OrderStatus.NEW
            )
            order_manager.orders[order.id] = order
            orders.append(order)
        
        with patch.object(order_manager, '_send_cancel_to_exchange', new_callable=AsyncMock) as mock_cancel:
            mock_cancel.return_value = True
            
            # Cancel all orders concurrently
            tasks = [order_manager.cancel_order(order.id) for order in orders]
            results = await asyncio.gather(*tasks)
            
            # Verify all cancellations succeeded
            assert all(results)
            
            # Verify all orders are cancelled
            for order in orders:
                assert order.status == OrderStatus.CANCELLED


class TestOrderMetrics:
    """Test order performance metrics and analytics."""

    @pytest.mark.asyncio
    async def test_fill_rate_calculation(self, order_manager):
        """Test calculation of order fill rates."""
        # Create orders with different outcomes
        filled_order = Order(
            id="filled", symbol="BTC/USDT", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=Decimal("1.0"),
            price=Decimal("50000.0"), status=OrderStatus.FILLED
        )
        cancelled_order = Order(
            id="cancelled", symbol="BTC/USDT", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=Decimal("1.0"),
            price=Decimal("50000.0"), status=OrderStatus.CANCELLED
        )
        
        order_manager.orders[filled_order.id] = filled_order
        order_manager.orders[cancelled_order.id] = cancelled_order
        
        fill_rate = order_manager.calculate_fill_rate("BTC/USDT")
        
        assert fill_rate == 0.5  # 1 filled out of 2 total

    @pytest.mark.asyncio
    async def test_average_fill_time(self, order_manager):
        """Test calculation of average fill time."""
        now = datetime.now(timezone.utc)
        
        # Create filled order with timing
        order = Order(
            id="timed", symbol="BTC/USDT", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=Decimal("1.0"),
            price=Decimal("50000.0"), status=OrderStatus.FILLED,
            created_at=now, filled_at=now
        )
        
        order_manager.orders[order.id] = order
        
        avg_time = order_manager.calculate_average_fill_time("BTC/USDT")
        
        assert avg_time is not None
        assert isinstance(avg_time, float)