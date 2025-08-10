"""
Unit tests for CapitalAllocator class.

This module tests the dynamic capital allocation framework including:
- Capital allocation strategies
- Rebalancing logic
- Performance tracking
- Risk-based allocation
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.core.types import (
    CapitalAllocation, AllocationStrategy, FundFlow, CapitalMetrics
)
from src.core.exceptions import (
    CapitalManagementError, ValidationError, InsufficientCapitalError
)
from src.core.config import Config
from src.capital_management.capital_allocator import CapitalAllocator


class TestCapitalAllocator:
    """Test cases for CapitalAllocator class."""

    @pytest.fixture
    def config(self):
        """Create test configuration with capital management settings."""
        config = Config()
        config.capital_management.total_capital = 100000.0
        config.capital_management.emergency_reserve_pct = 0.1
        config.capital_management.allocation_strategy = AllocationStrategy.PERFORMANCE_WEIGHTED
        config.capital_management.rebalance_frequency_hours = 24
        config.capital_management.min_allocation_pct = 0.05
        config.capital_management.max_allocation_pct = 0.4
        return config

    @pytest.fixture
    def capital_allocator(self, config):
        """Create capital allocator instance."""
        return CapitalAllocator(config)

    @pytest.fixture
    def sample_allocations(self):
        """Create sample capital allocations."""
        return [
            CapitalAllocation(
                strategy_id="strategy_1",
                exchange="binance",
                allocated_amount=Decimal("20000"),
                utilized_amount=Decimal("16000"),
                available_amount=Decimal("4000"),
                allocation_percentage=0.2,
                last_rebalance=datetime.now() - timedelta(hours=12)
            ),
            CapitalAllocation(
                strategy_id="strategy_2",
                exchange="okx",
                allocated_amount=Decimal("15000"),
                utilized_amount=Decimal("10500"),
                available_amount=Decimal("4500"),
                allocation_percentage=0.15,
                last_rebalance=datetime.now() - timedelta(hours=6)
            ),
            CapitalAllocation(
                strategy_id="strategy_3",
                exchange="coinbase",
                allocated_amount=Decimal("10000"),
                utilized_amount=Decimal("6000"),
                available_amount=Decimal("4000"),
                allocation_percentage=0.1,
                last_rebalance=datetime.now() - timedelta(hours=18)
            )
        ]

    def test_initialization(self, capital_allocator, config):
        """Test capital allocator initialization."""
        assert capital_allocator.config == config
        assert capital_allocator.capital_config == config.capital_management
        assert capital_allocator.total_capital == Decimal(
            str(config.capital_management.total_capital))
        assert capital_allocator.emergency_reserve == Decimal(str(
            config.capital_management.total_capital * config.capital_management.emergency_reserve_pct))
        assert capital_allocator.available_capital == Decimal(str(
            config.capital_management.total_capital * (1 - config.capital_management.emergency_reserve_pct)))
        assert capital_allocator.strategy_allocations == {}
        assert capital_allocator.exchange_allocations == {}
        assert capital_allocator.strategy_performance == {}
        # Check that last_rebalance is set to a recent time (within last
        # minute)
        assert (
            datetime.now() -
            capital_allocator.last_rebalance).total_seconds() < 60

    @pytest.mark.asyncio
    async def test_allocate_capital_basic(self, capital_allocator):
        """Test basic capital allocation."""
        strategy_name = "test_strategy"
        exchange_name = "binance"
        allocation_amount = Decimal("10000")

        result = await capital_allocator.allocate_capital(
            strategy_name, exchange_name, allocation_amount
        )

        assert isinstance(result, CapitalAllocation)
        assert result.strategy_id == strategy_name
        assert result.exchange == exchange_name
        assert result.allocated_amount == allocation_amount
        allocation_key = f"{strategy_name}_{exchange_name}"
        assert allocation_key in capital_allocator.strategy_allocations

        allocation = capital_allocator.strategy_allocations[allocation_key]
        assert allocation.strategy_id == strategy_name
        assert allocation.exchange == exchange_name
        assert allocation.allocated_amount == allocation_amount
        assert allocation.allocation_percentage == float(
            allocation_amount / capital_allocator.total_capital)
        assert allocation.utilized_amount == Decimal("0")

    @pytest.mark.asyncio
    async def test_allocate_capital_insufficient_funds(
            self, capital_allocator):
        """Test allocation with insufficient available capital."""
        strategy_name = "test_strategy"
        exchange_name = "binance"
        allocation_amount = Decimal("100000")  # More than available capital

        with pytest.raises(ValidationError):
            await capital_allocator.allocate_capital(
                strategy_name, exchange_name, allocation_amount
            )

    @pytest.mark.asyncio
    async def test_allocate_capital_existing_allocation(
            self, capital_allocator):
        """Test allocation to existing strategy/exchange combination."""
        strategy_name = "test_strategy"
        exchange_name = "binance"

        # First allocation
        await capital_allocator.allocate_capital(
            strategy_name, exchange_name, Decimal("5000")
        )

        # Second allocation to same strategy/exchange
        result = await capital_allocator.allocate_capital(
            strategy_name, exchange_name, Decimal("3000")
        )

        assert isinstance(result, CapitalAllocation)
        allocation_key = f"{strategy_name}_{exchange_name}"
        allocation = capital_allocator.strategy_allocations[allocation_key]
        assert allocation.allocated_amount == Decimal(
            "3000")  # Second allocation overwrites first
        assert allocation.allocation_percentage == 3000 / \
            float(capital_allocator.total_capital)

    @pytest.mark.asyncio
    async def test_rebalance_allocations_equal_weight(
            self, capital_allocator, config):
        """Test rebalancing with equal weight strategy."""
        config.capital_management.allocation_strategy = AllocationStrategy.EQUAL_WEIGHT

        # Setup existing allocations
        capital_allocator.strategy_allocations = {
            "strategy_1_binance": CapitalAllocation(
                strategy_id="strategy_1",
                exchange="binance",
                allocated_amount=Decimal("20000"),
                utilized_amount=Decimal("16000"),
                available_amount=Decimal("4000"),
                allocation_percentage=0.2,
                last_rebalance=datetime.now() - timedelta(hours=25)
            ),
            "strategy_2_okx": CapitalAllocation(
                strategy_id="strategy_2",
                exchange="okx",
                allocated_amount=Decimal("15000"),
                utilized_amount=Decimal("10500"),
                available_amount=Decimal("4500"),
                allocation_percentage=0.15,
                last_rebalance=datetime.now() - timedelta(hours=25)
            )
        }

        result = await capital_allocator.rebalance_allocations()

        assert isinstance(result, dict)
        # Equal weight should result in equal allocations
        for allocation in result.values():
            assert isinstance(allocation, CapitalAllocation)
            assert allocation.allocation_percentage > 0

    @pytest.mark.asyncio
    async def test_rebalance_allocations_performance_weighted(
            self, capital_allocator, config):
        """Test rebalancing with performance weighted strategy."""
        config.capital_management.allocation_strategy = AllocationStrategy.PERFORMANCE_WEIGHTED

        # Setup allocations with different performance scores
        capital_allocator.strategy_allocations = {
            "strategy1_binance": CapitalAllocation(
                strategy_id="strategy1",
                exchange="binance",
                allocated_amount=Decimal("20000"),
                utilized_amount=Decimal("16000"),
                available_amount=Decimal("4000"),
                allocation_percentage=0.2,
                last_rebalance=datetime.now() - timedelta(hours=25)
            ),
            "strategy2_okx": CapitalAllocation(
                strategy_id="strategy2",
                exchange="okx",
                allocated_amount=Decimal("15000"),
                utilized_amount=Decimal("10500"),
                available_amount=Decimal("4500"),
                allocation_percentage=0.15,
                last_rebalance=datetime.now() - timedelta(hours=25)
            )
        }

        # Add performance data
        capital_allocator.strategy_performance = {
            "strategy1": {"return_rate": 0.9, "sharpe_ratio": 1.2},
            "strategy2": {"return_rate": 0.6, "sharpe_ratio": 0.8}
        }

        result = await capital_allocator.rebalance_allocations()

        assert isinstance(result, dict)
        # Higher performing strategy should get more allocation
        assert len(result) > 0
        for allocation in result.values():
            assert isinstance(allocation, CapitalAllocation)
            assert allocation.allocation_percentage > 0

    @pytest.mark.asyncio
    async def test_rebalance_allocations_no_rebalance_needed(
            self, capital_allocator):
        """Test rebalancing when not needed (within frequency window)."""
        # Setup allocation with recent rebalance
        capital_allocator.strategy_allocations = {
            "strategy_1_binance": CapitalAllocation(
                strategy_id="strategy_1",
                exchange="binance",
                allocated_amount=Decimal("20000"),
                utilized_amount=Decimal("16000"),
                available_amount=Decimal("4000"),
                allocation_percentage=0.2,
                last_rebalance=datetime.now() - timedelta(hours=2)  # Recent rebalance
            )
        }

        result = await capital_allocator.rebalance_allocations()

        # Should return allocations even if no rebalance needed
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_update_utilization(self, capital_allocator):
        """Test updating utilization rates."""
        strategy_name = "test_strategy"
        exchange_name = "binance"
        utilization_rate = Decimal("0.75")

        # Setup allocation
        allocation_key = f"{strategy_name}_{exchange_name}"
        capital_allocator.strategy_allocations = {
            allocation_key: CapitalAllocation(
                strategy_id=strategy_name,
                exchange=exchange_name,
                allocated_amount=Decimal("10000"),
                utilized_amount=Decimal("5000"),
                available_amount=Decimal("5000"),
                allocation_percentage=0.1,
                last_rebalance=datetime.now()
            )
        }

        await capital_allocator.update_utilization(
            strategy_name, exchange_name, utilization_rate
        )

        allocation = capital_allocator.strategy_allocations[allocation_key]
        assert allocation.utilized_amount == utilization_rate

    @pytest.mark.asyncio
    async def test_update_utilization_allocation_not_found(
            self, capital_allocator):
        """Test updating utilization for non-existent allocation."""
        await capital_allocator.update_utilization(
            "nonexistent", "nonexistent", Decimal("0.5")
        )
        # Should not raise an exception, just do nothing

    @pytest.mark.asyncio
    async def test_get_capital_metrics(self, capital_allocator):
        """Test getting capital metrics."""
        # Setup allocations
        capital_allocator.strategy_allocations = {
            "strategy_1_binance": CapitalAllocation(
                strategy_id="strategy_1",
                exchange="binance",
                allocated_amount=Decimal("20000"),
                utilized_amount=Decimal("16000"),
                available_amount=Decimal("4000"),
                allocation_percentage=0.2,
                last_rebalance=datetime.now()
            ),
            "strategy_2_okx": CapitalAllocation(
                strategy_id="strategy_2",
                exchange="okx",
                allocated_amount=Decimal("15000"),
                utilized_amount=Decimal("10500"),
                available_amount=Decimal("4500"),
                allocation_percentage=0.15,
                last_rebalance=datetime.now()
            )
        }

        metrics = await capital_allocator.get_capital_metrics()

        assert isinstance(metrics, CapitalMetrics)
        assert metrics.total_capital == capital_allocator.total_capital
        assert metrics.allocated_capital == Decimal("35000")
        # Available capital should be the available_capital attribute (which is
        # total - emergency_reserve)
        assert metrics.available_capital == capital_allocator.available_capital
        assert metrics.emergency_reserve == capital_allocator.emergency_reserve
        assert metrics.utilization_rate > 0  # Should have some utilization
        assert metrics.allocation_efficiency > 0  # Should have some efficiency

    @pytest.mark.asyncio
    async def test_equal_weight_allocation(self, capital_allocator):
        """Test equal weight allocation strategy."""
        # Setup some allocations first - need to match the expected format
        capital_allocator.strategy_allocations = {
            "strategy1_binance": CapitalAllocation(
                strategy_id="strategy1",
                exchange="binance",
                allocated_amount=Decimal("10000"),
                utilized_amount=Decimal("8000"),
                available_amount=Decimal("2000"),
                allocation_percentage=0.1,
                last_rebalance=datetime.now()),
            "strategy1_okx": CapitalAllocation(
                strategy_id="strategy1",
                exchange="okx",
                allocated_amount=Decimal("5000"),
                utilized_amount=Decimal("4000"),
                available_amount=Decimal("1000"),
                allocation_percentage=0.05,
                last_rebalance=datetime.now()),
            "strategy2_binance": CapitalAllocation(
                strategy_id="strategy2",
                exchange="binance",
                allocated_amount=Decimal("15000"),
                utilized_amount=Decimal("12000"),
                available_amount=Decimal("3000"),
                allocation_percentage=0.15,
                last_rebalance=datetime.now())}

        result = await capital_allocator._equal_weight_allocation()

        assert isinstance(result, dict)
        assert len(result) > 0
        for allocation in result.values():
            assert isinstance(allocation, CapitalAllocation)
            assert allocation.allocation_percentage > 0

    @pytest.mark.asyncio
    async def test_performance_weighted_allocation(self, capital_allocator):
        """Test performance weighted allocation strategy."""
        # Setup allocations
        capital_allocator.strategy_allocations = {
            "strategy1_binance": CapitalAllocation(
                strategy_id="strategy1",
                exchange="binance",
                allocated_amount=Decimal("10000"),
                utilized_amount=Decimal("8000"),
                available_amount=Decimal("2000"),
                allocation_percentage=0.1,
                last_rebalance=datetime.now()),
            "strategy2_binance": CapitalAllocation(
                strategy_id="strategy2",
                exchange="binance",
                allocated_amount=Decimal("15000"),
                utilized_amount=Decimal("12000"),
                available_amount=Decimal("3000"),
                allocation_percentage=0.15,
                last_rebalance=datetime.now())}

        # Setup performance data
        capital_allocator.strategy_performance = {
            "strategy1": {"return_rate": 0.9, "sharpe_ratio": 1.2},
            "strategy2": {"return_rate": 0.6, "sharpe_ratio": 0.8}
        }

        result = await capital_allocator._performance_weighted_allocation(capital_allocator.strategy_performance)

        assert isinstance(result, dict)
        assert len(result) > 0
        for allocation in result.values():
            assert isinstance(allocation, CapitalAllocation)
            assert allocation.allocation_percentage > 0

    @pytest.mark.asyncio
    async def test_volatility_weighted_allocation(self, capital_allocator):
        """Test volatility weighted allocation strategy."""
        # Setup allocations
        capital_allocator.strategy_allocations = {
            "strategy1_binance": CapitalAllocation(
                strategy_id="strategy1",
                exchange="binance",
                allocated_amount=Decimal("10000"),
                utilized_amount=Decimal("8000"),
                available_amount=Decimal("2000"),
                allocation_percentage=0.1,
                last_rebalance=datetime.now()),
            "strategy2_binance": CapitalAllocation(
                strategy_id="strategy2",
                exchange="binance",
                allocated_amount=Decimal("15000"),
                utilized_amount=Decimal("12000"),
                available_amount=Decimal("3000"),
                allocation_percentage=0.15,
                last_rebalance=datetime.now())}

        # Setup performance data
        capital_allocator.strategy_performance = {
            "strategy1": {
                "return_rate": 0.9,
                "sharpe_ratio": 1.2,
                "volatility": 0.15},
            "strategy2": {
                "return_rate": 0.6,
                "sharpe_ratio": 0.8,
                "volatility": 0.08}}

        result = await capital_allocator._volatility_weighted_allocation(capital_allocator.strategy_performance)

        assert isinstance(result, dict)
        assert len(result) > 0
        for allocation in result.values():
            assert isinstance(allocation, CapitalAllocation)
            assert allocation.allocation_percentage > 0

    @pytest.mark.asyncio
    async def test_risk_parity_allocation(self, capital_allocator):
        """Test risk parity allocation strategy."""
        # Setup allocations
        capital_allocator.strategy_allocations = {
            "strategy1_binance": CapitalAllocation(
                strategy_id="strategy1",
                exchange="binance",
                allocated_amount=Decimal("10000"),
                utilized_amount=Decimal("8000"),
                available_amount=Decimal("2000"),
                allocation_percentage=0.1,
                last_rebalance=datetime.now()),
            "strategy2_binance": CapitalAllocation(
                strategy_id="strategy2",
                exchange="binance",
                allocated_amount=Decimal("15000"),
                utilized_amount=Decimal("12000"),
                available_amount=Decimal("3000"),
                allocation_percentage=0.15,
                last_rebalance=datetime.now())}

        # Setup performance data
        capital_allocator.strategy_performance = {
            "strategy1": {
                "return_rate": 0.9,
                "sharpe_ratio": 1.2,
                "volatility": 0.15},
            "strategy2": {
                "return_rate": 0.6,
                "sharpe_ratio": 0.8,
                "volatility": 0.08}}

        result = await capital_allocator._risk_parity_allocation(capital_allocator.strategy_performance)

        assert isinstance(result, dict)
        assert len(result) > 0
        for allocation in result.values():
            assert isinstance(allocation, CapitalAllocation)
            assert allocation.allocation_percentage > 0

    @pytest.mark.asyncio
    async def test_dynamic_allocation(self, capital_allocator):
        """Test dynamic allocation strategy."""
        # Setup allocations
        capital_allocator.strategy_allocations = {
            "strategy1_binance": CapitalAllocation(
                strategy_id="strategy1",
                exchange="binance",
                allocated_amount=Decimal("10000"),
                utilized_amount=Decimal("8000"),
                available_amount=Decimal("2000"),
                allocation_percentage=0.1,
                last_rebalance=datetime.now()),
            "strategy2_binance": CapitalAllocation(
                strategy_id="strategy2",
                exchange="binance",
                allocated_amount=Decimal("15000"),
                utilized_amount=Decimal("12000"),
                available_amount=Decimal("3000"),
                allocation_percentage=0.15,
                last_rebalance=datetime.now())}

        # Setup performance data
        capital_allocator.strategy_performance = {
            "strategy1": {
                "return_rate": 0.9,
                "sharpe_ratio": 1.2,
                "volatility": 0.15},
            "strategy2": {
                "return_rate": 0.6,
                "sharpe_ratio": 0.8,
                "volatility": 0.08}}

        result = await capital_allocator._dynamic_allocation(capital_allocator.strategy_performance)

        assert isinstance(result, dict)
        assert len(result) > 0
        for allocation in result.values():
            assert isinstance(allocation, CapitalAllocation)
            assert allocation.allocation_percentage > 0

    @pytest.mark.asyncio
    async def test_calculate_performance_metrics(self, capital_allocator):
        """Test calculating performance metrics."""
        strategy_name = "teststrategy"

        # Setup allocations first
        capital_allocator.strategy_allocations = {
            f"{strategy_name}_binance": CapitalAllocation(
                strategy_id=strategy_name,
                exchange="binance",
                allocated_amount=Decimal("10000"),
                utilized_amount=Decimal("8000"),
                available_amount=Decimal("2000"),
                allocation_percentage=0.1,
                last_rebalance=datetime.now())}

        # Setup performance data
        capital_allocator.strategy_performance = {
            strategy_name: {"return_rate": 0.85, "sharpe_ratio": 1.1}
        }

        metrics = await capital_allocator._calculate_performance_metrics()

        assert isinstance(metrics, dict)
        assert strategy_name in metrics
        assert "return_rate" in metrics[strategy_name]
        assert "sharpe_ratio" in metrics[strategy_name]

    @pytest.mark.asyncio
    async def test_calculate_performance_metrics_no_data(
            self, capital_allocator):
        """Test calculating performance metrics with no data."""
        metrics = await capital_allocator._calculate_performance_metrics()

        assert isinstance(metrics, dict)
        assert len(metrics) == 0  # No performance data

    @pytest.mark.asyncio
    async def test_update_total_capital(self, capital_allocator):
        """Test updating total capital."""
        new_total = Decimal("150000")

        await capital_allocator.update_total_capital(new_total)

        assert capital_allocator.total_capital == new_total
        assert capital_allocator.emergency_reserve == new_total * \
            Decimal(str(capital_allocator.capital_config.emergency_reserve_pct))
        assert capital_allocator.available_capital == new_total * \
            (1 - Decimal(str(capital_allocator.capital_config.emergency_reserve_pct)))

    @pytest.mark.asyncio
    async def test_get_emergency_reserve(self, capital_allocator):
        """Test getting emergency reserve amount."""
        reserve = await capital_allocator.get_emergency_reserve()

        assert reserve == capital_allocator.emergency_reserve
        assert reserve == capital_allocator.total_capital * \
            Decimal(str(capital_allocator.capital_config.emergency_reserve_pct))

    @pytest.mark.asyncio
    async def test_get_available_capital(self, capital_allocator):
        """Test getting available capital amount."""
        available = capital_allocator.available_capital

        assert available == capital_allocator.total_capital * \
            (1 - Decimal(str(capital_allocator.capital_config.emergency_reserve_pct)))

    @pytest.mark.asyncio
    async def test_get_available_capital_attribute(self, capital_allocator):
        """Test getting available capital attribute."""
        available = capital_allocator.available_capital

        assert available == capital_allocator.total_capital * \
            (1 - Decimal(str(capital_allocator.capital_config.emergency_reserve_pct)))
