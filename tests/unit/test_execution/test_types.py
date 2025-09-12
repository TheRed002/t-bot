"""Optimized unit tests for execution types."""

import logging
from decimal import Decimal

import pytest

# Disable logging for performance
logging.disable(logging.CRITICAL)

from src.core.types.execution import ExecutionAlgorithm, ExecutionStatus, SlippageType
from src.execution.types import ExecutionInstruction

# Pre-defined constants for faster test data creation
TEST_DECIMALS = {
    "ONE": Decimal("1.0"),
    "TWO_FIVE": Decimal("2.5"),
    "PRICE_50K": Decimal("50000"),
    "PRICE_2K": Decimal("2000.0")
}

# Cache common test configurations
COMMON_ATTRS = {
    "btc_symbol": "BTC/USDT",
    "eth_symbol": "ETH/USDT",
    "strategy_1": "test_strategy",
    "strategy_2": "test_strategy_2",
    "time_horizon_60": 60,
    "time_horizon_30": 30,
    "slices_10": 10,
    "slices_5": 5,
    "participation_rate": 0.25,
    "min_algorithms": 8,
    "min_statuses": 6
}


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
        """Test ExecutionInstruction creation with cached constants."""
        from src.core.types import OrderRequest, OrderSide, OrderType

        order = OrderRequest(
            symbol=COMMON_ATTRS["btc_symbol"],
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=TEST_DECIMALS["ONE"],
            price=TEST_DECIMALS["PRICE_50K"],
        )

        instruction = ExecutionInstruction(
            order=order,
            algorithm=ExecutionAlgorithm.TWAP,
            strategy_name=COMMON_ATTRS["strategy_1"],
            time_horizon_minutes=COMMON_ATTRS["time_horizon_60"],
            max_slices=COMMON_ATTRS["slices_10"],
        )

        assert instruction.order.symbol == COMMON_ATTRS["btc_symbol"]
        assert instruction.order.side == OrderSide.BUY
        assert instruction.order.quantity == TEST_DECIMALS["ONE"]
        assert instruction.algorithm == ExecutionAlgorithm.TWAP
        assert instruction.strategy_name == COMMON_ATTRS["strategy_1"]

    def test_execution_instruction_with_optional_fields(self):
        """Test ExecutionInstruction with optional fields using cached constants."""
        from src.core.types import OrderRequest, OrderSide, OrderType

        order = OrderRequest(
            symbol=COMMON_ATTRS["eth_symbol"],
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=TEST_DECIMALS["TWO_FIVE"],
            price=TEST_DECIMALS["PRICE_2K"],
        )

        instruction = ExecutionInstruction(
            order=order,
            algorithm=ExecutionAlgorithm.VWAP,
            strategy_name=COMMON_ATTRS["strategy_2"],
            time_horizon_minutes=COMMON_ATTRS["time_horizon_30"],
            max_slices=COMMON_ATTRS["slices_5"],
            participation_rate=COMMON_ATTRS["participation_rate"],
        )

        assert instruction.order.price == TEST_DECIMALS["PRICE_2K"]
        assert instruction.participation_rate == COMMON_ATTRS["participation_rate"]

    def test_enum_iteration(self):
        """Test that enums can be iterated."""
        algorithms = list(ExecutionAlgorithm)
        assert len(algorithms) >= COMMON_ATTRS["min_algorithms"]  # We have 8 algorithms now
        assert ExecutionAlgorithm.TWAP in algorithms

        statuses = list(ExecutionStatus)
        assert len(statuses) >= COMMON_ATTRS["min_statuses"]
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
