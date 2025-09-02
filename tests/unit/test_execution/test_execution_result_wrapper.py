"""Unit tests for ExecutionResultWrapper."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.execution.execution_result_wrapper import ExecutionResultWrapper
from src.core.types import ExecutionResult, ExecutionStatus, OrderRequest, OrderSide, OrderType, ExecutionAlgorithm


class TestExecutionResultWrapper:
    """Test cases for ExecutionResultWrapper."""

    @pytest.fixture
    def sample_execution_result(self):
        """Create sample execution result."""
        return ExecutionResult(
            instruction_id="test_001",
            symbol="BTC/USDT", 
            status=ExecutionStatus.COMPLETED,
            target_quantity=Decimal("1.0"),
            filled_quantity=Decimal("1.0"),
            remaining_quantity=Decimal("0.0"),
            average_price=Decimal("50000.0"),
            worst_price=Decimal("50100.0"),
            best_price=Decimal("49900.0"),
            expected_cost=Decimal("50000.0"),
            actual_cost=Decimal("50000.0"),
            slippage_bps=10.0,
            slippage_amount=Decimal("100.0"),
            fill_rate=1.0,
            execution_time=30,
            num_fills=5,
            num_orders=2,
            total_fees=Decimal("25.0"),
            maker_fees=Decimal("10.0"),
            taker_fees=Decimal("15.0"),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc)
        )

    @pytest.fixture
    def sample_order_request(self):
        """Create sample order request."""
        return OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
            client_order_id="test_order_001"
        )

    def test_execution_result_wrapper_creation(self, sample_execution_result, sample_order_request):
        """Test ExecutionResultWrapper creation."""
        wrapper = ExecutionResultWrapper(
            sample_execution_result,
            sample_order_request, 
            ExecutionAlgorithm.TWAP
        )
        
        assert wrapper.result == sample_execution_result
        assert wrapper.original_request == sample_order_request
        assert wrapper.algorithm == ExecutionAlgorithm.TWAP

    def test_wrapper_get_summary(self, sample_execution_result, sample_order_request):
        """Test getting execution summary."""
        wrapper = ExecutionResultWrapper(
            sample_execution_result,
            sample_order_request,
            ExecutionAlgorithm.TWAP
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
            sample_execution_result,
            sample_order_request,
            ExecutionAlgorithm.TWAP
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
            completed_at=None
        )
        
        wrapper = ExecutionResultWrapper(
            partial_result,
            sample_order_request,
            ExecutionAlgorithm.TWAP
        )
        
        assert wrapper.is_partial()

    def test_wrapper_get_performance_metrics(self, sample_execution_result, sample_order_request):
        """Test performance metrics calculation."""
        wrapper = ExecutionResultWrapper(
            sample_execution_result,
            sample_order_request,
            ExecutionAlgorithm.TWAP
        )
        
        metrics = wrapper.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert "slippage_bps" in metrics
        assert "fill_rate" in metrics
        assert "execution_time" in metrics

    def test_wrapper_calculate_efficiency(self, sample_execution_result, sample_order_request):
        """Test efficiency calculation."""
        wrapper = ExecutionResultWrapper(
            sample_execution_result,
            sample_order_request,
            ExecutionAlgorithm.TWAP
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
            completed_at=datetime.now(timezone.utc)
        )
        
        wrapper = ExecutionResultWrapper(
            failed_result,
            sample_order_request,
            ExecutionAlgorithm.TWAP
        )
        
        assert not wrapper.is_successful()
        assert wrapper.result.status == ExecutionStatus.FAILED