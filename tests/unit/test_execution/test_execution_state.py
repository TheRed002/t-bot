"""Optimized unit tests for execution state management."""

import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

# Disable logging for performance
logging.disable(logging.CRITICAL)

from src.core.types import OrderRequest, OrderResponse, OrderSide, OrderType, OrderStatus
from src.core.types.execution import ExecutionAlgorithm, ExecutionStatus
from src.execution.execution_state import ExecutionState

# Create fixed datetime instances to avoid repeated datetime.now() calls
FIXED_DATETIME = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
FIXED_DATETIME_PLUS_5 = FIXED_DATETIME + timedelta(seconds=5)
FIXED_DATETIME_PLUS_10 = FIXED_DATETIME + timedelta(seconds=10)

# Pre-defined constants for faster test data creation
TEST_DECIMALS = {
    "ZERO": Decimal("0"),
    "POINT_THREE": Decimal("0.3"),
    "POINT_SEVEN": Decimal("0.7"),
    "ONE": Decimal("1.0"),
    "PRICE_50K": Decimal("50000"),
    "PRICE_50K_PRECISE": Decimal("50000.0"),
    "PRICE_50100": Decimal("50100.0")
}

# Cache common test configurations
COMMON_ATTRS = {
    "symbol": "BTC/USDT",
    "execution_id": "test_001",
    "child_001": "child_001",
    "child_002": "child_002",
    "client_001": "client_001",
    "client_002": "client_002",
    "exchange": "test_exchange",
    "error_msg": "connection_error",
    "duration_5": 5.0,
    "duration_10": 10.0
}


class TestExecutionState:
    """Test cases for ExecutionState."""

    def test_execution_state_creation(self):
        """Test ExecutionState creation with cached constants."""
        order = OrderRequest(
            symbol=COMMON_ATTRS["symbol"],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=TEST_DECIMALS["ONE"],
            price=TEST_DECIMALS["PRICE_50K"],
        )

        state = ExecutionState(
            execution_id=COMMON_ATTRS["execution_id"],
            original_order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.PENDING,
            start_time=FIXED_DATETIME,
        )

        assert state.execution_id == COMMON_ATTRS["execution_id"]
        assert state.original_order.symbol == COMMON_ATTRS["symbol"]
        assert state.original_order.quantity == TEST_DECIMALS["ONE"]
        assert state.algorithm == ExecutionAlgorithm.TWAP
        assert state.status == ExecutionStatus.PENDING
        assert state.total_filled_quantity == TEST_DECIMALS["ZERO"]

    def test_execution_state_add_child_order(self):
        """Test adding child order and updating metrics with cached constants."""
        order = OrderRequest(
            symbol=COMMON_ATTRS["symbol"],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=TEST_DECIMALS["ONE"],
            price=TEST_DECIMALS["PRICE_50K"],
        )

        state = ExecutionState(
            execution_id=COMMON_ATTRS["execution_id"],
            original_order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.PENDING,
            start_time=FIXED_DATETIME,
        )

        # Create a child order response with cached values
        child_order = OrderResponse(
            id=COMMON_ATTRS["child_001"],
            client_order_id=COMMON_ATTRS["client_001"],
            symbol=COMMON_ATTRS["symbol"],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=TEST_DECIMALS["POINT_THREE"],
            price=TEST_DECIMALS["PRICE_50K_PRECISE"],
            filled_quantity=TEST_DECIMALS["POINT_THREE"],
            status=OrderStatus.FILLED,
            created_at=FIXED_DATETIME,
            exchange=COMMON_ATTRS["exchange"],
        )

        # Add child order
        state.add_child_order(child_order)

        assert state.total_filled_quantity == TEST_DECIMALS["POINT_THREE"]
        assert len(state.child_orders) == 1
        assert state.average_fill_price == TEST_DECIMALS["PRICE_50K_PRECISE"]
        assert state.number_of_trades == 1

    def test_execution_state_multiple_fills(self):
        """Test execution state with multiple fills using cached constants."""
        order = OrderRequest(
            symbol=COMMON_ATTRS["symbol"],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=TEST_DECIMALS["ONE"],
            price=TEST_DECIMALS["PRICE_50K"],
        )

        state = ExecutionState(
            execution_id=COMMON_ATTRS["execution_id"],
            original_order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.PENDING,
            start_time=FIXED_DATETIME,
        )

        # Add first fill with cached values
        child_order1 = OrderResponse(
            id=COMMON_ATTRS["child_001"],
            client_order_id=COMMON_ATTRS["client_001"],
            symbol=COMMON_ATTRS["symbol"],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=TEST_DECIMALS["POINT_THREE"],
            price=TEST_DECIMALS["PRICE_50K_PRECISE"],
            filled_quantity=TEST_DECIMALS["POINT_THREE"],
            status=OrderStatus.FILLED,
            created_at=FIXED_DATETIME,
            exchange=COMMON_ATTRS["exchange"],
        )
        state.add_child_order(child_order1)

        # Add second fill with cached values
        child_order2 = OrderResponse(
            id=COMMON_ATTRS["child_002"],
            client_order_id=COMMON_ATTRS["client_002"],
            symbol=COMMON_ATTRS["symbol"],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=TEST_DECIMALS["POINT_SEVEN"],
            price=TEST_DECIMALS["PRICE_50100"],
            filled_quantity=TEST_DECIMALS["POINT_SEVEN"],
            status=OrderStatus.FILLED,
            created_at=FIXED_DATETIME,
            exchange=COMMON_ATTRS["exchange"],
        )
        state.add_child_order(child_order2)

        assert state.total_filled_quantity == TEST_DECIMALS["ONE"]
        assert len(state.child_orders) == 2
        assert state.number_of_trades == 2

        # Check average price calculation using pre-defined constants
        expected_avg = (
            TEST_DECIMALS["POINT_THREE"] * TEST_DECIMALS["PRICE_50K_PRECISE"] +
            TEST_DECIMALS["POINT_SEVEN"] * TEST_DECIMALS["PRICE_50100"]
        ) / TEST_DECIMALS["ONE"]
        assert state.average_fill_price == expected_avg

    def test_execution_state_set_completed(self):
        """Test marking execution as completed with cached datetimes."""
        order = OrderRequest(
            symbol=COMMON_ATTRS["symbol"],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=TEST_DECIMALS["ONE"],
            price=TEST_DECIMALS["PRICE_50K"],
        )

        state = ExecutionState(
            execution_id=COMMON_ATTRS["execution_id"],
            original_order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.PENDING,
            start_time=FIXED_DATETIME,
        )

        state.set_completed(FIXED_DATETIME_PLUS_10)

        assert state.end_time == FIXED_DATETIME_PLUS_10
        assert state.execution_duration == COMMON_ATTRS["duration_10"]

    def test_execution_state_set_failed(self):
        """Test marking execution as failed with cached values."""
        order = OrderRequest(
            symbol=COMMON_ATTRS["symbol"],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=TEST_DECIMALS["ONE"],
            price=TEST_DECIMALS["PRICE_50K"],
        )

        state = ExecutionState(
            execution_id=COMMON_ATTRS["execution_id"],
            original_order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            status=ExecutionStatus.PENDING,
            start_time=FIXED_DATETIME,
        )

        state.set_failed(COMMON_ATTRS["error_msg"], FIXED_DATETIME_PLUS_5)

        assert state.status == ExecutionStatus.FAILED
        assert state.error_message == COMMON_ATTRS["error_msg"]
        assert state.end_time == FIXED_DATETIME_PLUS_5
        assert state.execution_duration == COMMON_ATTRS["duration_5"]
