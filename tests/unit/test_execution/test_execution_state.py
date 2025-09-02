"""Unit tests for execution state management."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.execution.execution_state import ExecutionState
from src.core.types.execution import ExecutionAlgorithm, ExecutionStatus
from src.core.types import OrderRequest, OrderResponse, OrderSide, OrderType


class TestExecutionState:
    """Test cases for ExecutionState."""

    def test_execution_state_creation(self):
        """Test ExecutionState creation."""
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000")
        )
        
        state = ExecutionState(
            execution_id="test_001",
            original_order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.PENDING,
            start_time=datetime.now(timezone.utc)
        )
        
        assert state.execution_id == "test_001"
        assert state.original_order.symbol == "BTC/USDT"
        assert state.original_order.quantity == Decimal("1.0")
        assert state.algorithm == ExecutionAlgorithm.TWAP
        assert state.status == ExecutionStatus.PENDING
        assert state.total_filled_quantity == Decimal("0")

    def test_execution_state_add_child_order(self):
        """Test adding child order and updating metrics."""
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000")
        )
        
        state = ExecutionState(
            execution_id="test_001",
            original_order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.PENDING,
            start_time=datetime.now(timezone.utc)
        )
        
        # Create a child order response
        from src.core.types import OrderStatus
        child_order = OrderResponse(
            id="child_001",
            client_order_id="client_001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.3"),
            price=Decimal("50000.0"),
            filled_quantity=Decimal("0.3"),
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            exchange="test_exchange"
        )
        
        # Add child order
        state.add_child_order(child_order)
        
        assert state.total_filled_quantity == Decimal("0.3")
        assert len(state.child_orders) == 1
        assert state.average_fill_price == Decimal("50000.0")
        assert state.number_of_trades == 1

    def test_execution_state_multiple_fills(self):
        """Test execution state with multiple fills."""
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000")
        )
        
        state = ExecutionState(
            execution_id="test_001",
            original_order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.PENDING,
            start_time=datetime.now(timezone.utc)
        )
        
        # Add first fill
        from src.core.types import OrderStatus
        child_order1 = OrderResponse(
            id="child_001",
            client_order_id="client_001",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.3"),
            price=Decimal("50000.0"),
            filled_quantity=Decimal("0.3"),
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            exchange="test_exchange"
        )
        state.add_child_order(child_order1)
        
        # Add second fill
        child_order2 = OrderResponse(
            id="child_002",
            client_order_id="client_002",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.7"),
            price=Decimal("50100.0"),
            filled_quantity=Decimal("0.7"),
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            exchange="test_exchange"
        )
        state.add_child_order(child_order2)
        
        assert state.total_filled_quantity == Decimal("1.0")
        assert len(state.child_orders) == 2
        assert state.number_of_trades == 2
        
        # Check average price calculation
        expected_avg = (Decimal("0.3") * Decimal("50000.0") + Decimal("0.7") * Decimal("50100.0")) / Decimal("1.0")
        assert state.average_fill_price == expected_avg

    def test_execution_state_set_completed(self):
        """Test marking execution as completed."""
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000")
        )
        
        start_time = datetime.now(timezone.utc)
        state = ExecutionState(
            execution_id="test_001",
            original_order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.PENDING,
            start_time=start_time
        )
        
        from datetime import timedelta
        end_time = start_time + timedelta(seconds=10)  # 10 seconds later
        state.set_completed(end_time)
        
        assert state.end_time == end_time
        assert state.execution_duration == 10.0

    def test_execution_state_set_failed(self):
        """Test marking execution as failed."""
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000")
        )
        
        start_time = datetime.now(timezone.utc)
        state = ExecutionState(
            execution_id="test_001",
            original_order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.PENDING,
            start_time=start_time
        )
        
        from datetime import timedelta
        end_time = start_time + timedelta(seconds=5)  # 5 seconds later
        state.set_failed("connection_error", end_time)
        
        assert state.status == ExecutionStatus.FAILED
        assert state.error_message == "connection_error"
        assert state.end_time == end_time
        assert state.execution_duration == 5.0