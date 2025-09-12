"""Unit tests for ExecutionResultWrapper."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest

# Disable logging for performance
logging.disable(logging.CRITICAL)

# Create fixed datetime instances to avoid repeated datetime.now() calls
FIXED_DATETIME = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# Pre-defined constants for faster test data creation
TEST_DECIMALS = {
    "ZERO": Decimal("0.0"),
    "ONE": Decimal("1.0"),
    "PRICE_50K": Decimal("50000.0"),
    "PRICE_49_9K": Decimal("49900.0"),
    "PRICE_50_1K": Decimal("50100.0"),
    "AMOUNT_25": Decimal("25.0"),
    "AMOUNT_10": Decimal("10.0"),
    "AMOUNT_15": Decimal("15.0"),
    "AMOUNT_100": Decimal("100.0"),
}

TEST_DATA = {
    "INSTRUCTION_ID": "test_001",
    "SYMBOL": "BTC/USDT",
    "CLIENT_ORDER_ID": "test_order_001",
    "EXECUTION_TIME": 30,
    "NUM_FILLS": 5,
    "NUM_ORDERS": 2,
    "SLIPPAGE_BPS": 10.0,
    "FILL_RATE": 1.0
}

from src.core.types import (
    ExecutionAlgorithm,
    ExecutionResult,
    ExecutionStatus,
    OrderRequest,
    OrderSide,
    OrderType,
)
from src.execution.execution_result_wrapper import ExecutionResultWrapper


class TestExecutionResultWrapper:
    """Test cases for ExecutionResultWrapper."""

    @pytest.fixture(scope="session")
    def sample_execution_result(self):
        """Create sample execution result with pre-defined constants."""
        return ExecutionResult(
            instruction_id=TEST_DATA["INSTRUCTION_ID"],
            symbol=TEST_DATA["SYMBOL"],
            status=ExecutionStatus.COMPLETED,
            target_quantity=TEST_DECIMALS["ONE"],
            filled_quantity=TEST_DECIMALS["ONE"],
            remaining_quantity=TEST_DECIMALS["ZERO"],
            average_price=TEST_DECIMALS["PRICE_50K"],
            worst_price=TEST_DECIMALS["PRICE_50_1K"],
            best_price=TEST_DECIMALS["PRICE_49_9K"],
            expected_cost=TEST_DECIMALS["PRICE_50K"],
            actual_cost=TEST_DECIMALS["PRICE_50K"],
            slippage_bps=TEST_DATA["SLIPPAGE_BPS"],
            slippage_amount=TEST_DECIMALS["AMOUNT_100"],
            fill_rate=TEST_DATA["FILL_RATE"],
            execution_time=TEST_DATA["EXECUTION_TIME"],
            num_fills=TEST_DATA["NUM_FILLS"],
            num_orders=TEST_DATA["NUM_ORDERS"],
            total_fees=TEST_DECIMALS["AMOUNT_25"],
            maker_fees=TEST_DECIMALS["AMOUNT_10"],
            taker_fees=TEST_DECIMALS["AMOUNT_15"],
            started_at=FIXED_DATETIME,
            completed_at=FIXED_DATETIME,
        )

    @pytest.fixture(scope="session")
    def sample_order_request(self):
        """Create sample order request with cached constants."""
        return OrderRequest(
            symbol=TEST_DATA["SYMBOL"],
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=TEST_DECIMALS["ONE"],
            price=TEST_DECIMALS["PRICE_50K"],
            client_order_id=TEST_DATA["CLIENT_ORDER_ID"],
        )

    def test_execution_result_wrapper_creation(self, sample_execution_result, sample_order_request):
        """Test ExecutionResultWrapper creation."""
        wrapper = ExecutionResultWrapper(
            sample_execution_result, sample_order_request, ExecutionAlgorithm.TWAP
        )

        assert wrapper.result == sample_execution_result
        assert wrapper.original_request == sample_order_request
        assert wrapper.algorithm == ExecutionAlgorithm.TWAP

    def test_wrapper_get_summary(self, sample_execution_result, sample_order_request):
        """Test getting execution summary."""
        wrapper = ExecutionResultWrapper(
            sample_execution_result, sample_order_request, ExecutionAlgorithm.TWAP
        )

        summary = wrapper.get_summary()
        assert isinstance(summary, dict)
        assert "instruction_id" in summary
        assert "symbol" in summary
        assert "status" in summary
        assert "filled_quantity" in summary

    def test_wrapper_is_successful(self, sample_execution_result, sample_order_request):
        """Test success check."""
        wrapper = ExecutionResultWrapper(
            sample_execution_result, sample_order_request, ExecutionAlgorithm.TWAP
        )

        # Should be successful since status is COMPLETED
        assert wrapper.is_successful()

    def test_wrapper_is_partial(self, sample_order_request):
        """Test partial execution check."""
        partial_result = ExecutionResult(
            instruction_id="test_001",
            symbol="BTC/USDT",
            status=ExecutionStatus.PARTIAL,
            target_quantity=Decimal("1.0"),
            filled_quantity=Decimal("0.5"),
            remaining_quantity=Decimal("0.5"),
            average_price=Decimal("50000.0"),
            worst_price=Decimal("50000.0"),
            best_price=Decimal("50000.0"),
            expected_cost=Decimal("50000.0"),
            actual_cost=Decimal("25000.0"),
            slippage_bps=0.0,
            slippage_amount=Decimal("0.0"),
            fill_rate=0.5,
            execution_time=15,
            num_fills=1,
            num_orders=1,
            total_fees=Decimal("12.5"),
            maker_fees=Decimal("12.5"),
            taker_fees=Decimal("0.0"),
            started_at=datetime.now(timezone.utc),
            completed_at=None,
        )

        wrapper = ExecutionResultWrapper(
            partial_result, sample_order_request, ExecutionAlgorithm.TWAP
        )

        assert wrapper.is_partial()

    def test_wrapper_get_performance_metrics(self, sample_execution_result, sample_order_request):
        """Test performance metrics calculation."""
        wrapper = ExecutionResultWrapper(
            sample_execution_result, sample_order_request, ExecutionAlgorithm.TWAP
        )

        metrics = wrapper.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert "slippage_bps" in metrics
        assert "fill_rate" in metrics
        assert "execution_time" in metrics

    def test_wrapper_calculate_efficiency(self, sample_execution_result, sample_order_request):
        """Test efficiency calculation."""
        wrapper = ExecutionResultWrapper(
            sample_execution_result, sample_order_request, ExecutionAlgorithm.TWAP
        )

        efficiency = wrapper.calculate_efficiency()
        assert isinstance(efficiency, (float, int, Decimal))
        assert efficiency >= 0

    def test_wrapper_with_failed_execution(self, sample_order_request):
        """Test wrapper with failed execution."""
        failed_result = ExecutionResult(
            instruction_id="test_001",
            symbol="BTC/USDT",
            status=ExecutionStatus.FAILED,
            target_quantity=Decimal("1.0"),
            filled_quantity=Decimal("0.0"),
            remaining_quantity=Decimal("1.0"),
            average_price=Decimal("0.0"),
            worst_price=Decimal("0.0"),
            best_price=Decimal("0.0"),
            expected_cost=Decimal("50000.0"),
            actual_cost=Decimal("0.0"),
            slippage_bps=0.0,
            slippage_amount=Decimal("0.0"),
            fill_rate=0.0,
            execution_time=0,
            num_fills=0,
            num_orders=0,
            total_fees=Decimal("0.0"),
            maker_fees=Decimal("0.0"),
            taker_fees=Decimal("0.0"),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        wrapper = ExecutionResultWrapper(
            failed_result, sample_order_request, ExecutionAlgorithm.TWAP
        )

        assert not wrapper.is_successful()
        assert wrapper.result.status == ExecutionStatus.FAILED
