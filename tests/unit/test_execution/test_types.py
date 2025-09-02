"""Unit tests for execution types."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

from src.execution.types import ExecutionInstruction
from src.core.types.execution import ExecutionAlgorithm, ExecutionStatus, SlippageType


class TestExecutionTypes:
    """Test cases for execution types."""

    def test_execution_algorithm_enum_values(self):
        """Test ExecutionAlgorithm enum values."""
        assert ExecutionAlgorithm.MARKET.value == "market"
        assert ExecutionAlgorithm.LIMIT.value == "limit"
        assert ExecutionAlgorithm.TWAP.value == "twap"
        assert ExecutionAlgorithm.VWAP.value == "vwap"
        assert ExecutionAlgorithm.ICEBERG.value == "iceberg"
        assert ExecutionAlgorithm.SMART_ROUTER.value == "smart_router"

    def test_execution_status_enum_values(self):
        """Test ExecutionStatus enum values."""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.CANCELLED.value == "cancelled"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.PARTIAL.value == "partial"

    def test_slippage_type_enum_values(self):
        """Test SlippageType enum values."""
        assert SlippageType.MARKET_IMPACT.value == "market_impact"
        assert SlippageType.TIMING.value == "timing"
        assert SlippageType.SPREAD.value == "spread"
        assert SlippageType.FEES.value == "fees"
        assert SlippageType.PRICE_IMPROVEMENT.value == "price_improvement"

    def test_execution_instruction_creation(self):
        """Test ExecutionInstruction creation."""
        from src.core.types import OrderRequest, OrderSide, OrderType
        
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000")
        )
        
        instruction = ExecutionInstruction(
            order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            strategy_name="test_strategy",
            time_horizon_minutes=60,
            max_slices=10
        )
        
        assert instruction.order.symbol == "BTC/USDT"
        assert instruction.order.side == OrderSide.BUY
        assert instruction.order.quantity == Decimal("1.0")
        assert instruction.algorithm == ExecutionAlgorithm.TWAP
        assert instruction.strategy_name == "test_strategy"

    def test_execution_instruction_with_optional_fields(self):
        """Test ExecutionInstruction with optional fields."""
        from src.core.types import OrderRequest, OrderSide, OrderType
        
        order = OrderRequest(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("2.5"),
            price=Decimal("2000.0")
        )
        
        instruction = ExecutionInstruction(
            order=order,
            algorithm=ExecutionAlgorithm.VWAP,
            strategy_name="test_strategy_2",
            time_horizon_minutes=30,
            max_slices=5,
            participation_rate=0.25
        )
        
        assert instruction.order.price == Decimal("2000.0")
        assert instruction.participation_rate == 0.25

    def test_enum_iteration(self):
        """Test that enums can be iterated."""
        algorithms = list(ExecutionAlgorithm)
        assert len(algorithms) >= 8  # We have 8 algorithms now
        assert ExecutionAlgorithm.TWAP in algorithms
        
        statuses = list(ExecutionStatus)
        assert len(statuses) >= 6
        assert ExecutionStatus.COMPLETED in statuses

    def test_enum_membership(self):
        """Test enum membership checks."""
        assert ExecutionAlgorithm.TWAP in ExecutionAlgorithm
        assert ExecutionStatus.COMPLETED in ExecutionStatus
        assert SlippageType.MARKET_IMPACT in SlippageType

    def test_enum_string_representation(self):
        """Test enum string representation."""
        assert str(ExecutionAlgorithm.TWAP) == "ExecutionAlgorithm.TWAP"
        assert str(ExecutionStatus.COMPLETED) == "ExecutionStatus.COMPLETED"
        
        # Test repr
        assert repr(ExecutionAlgorithm.TWAP) == "<ExecutionAlgorithm.TWAP: 'twap'>"