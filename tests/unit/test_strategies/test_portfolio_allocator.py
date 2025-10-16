"""
Test suite for PortfolioAllocator - tests only methods that actually exist.
"""

import logging
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import numpy as np

# Disable logging for performance
logging.disable(logging.CRITICAL)

from src.core.types import StrategyType, StrategyStatus, MarketRegime
from src.strategies.portfolio_allocator import PortfolioAllocator, StrategyAllocation
from src.risk_management.base import BaseRiskManager
from src.strategies.interfaces import BaseStrategyInterface


@pytest.fixture(scope="session")
def mock_strategy():
    """Mock strategy interface."""
    strategy = Mock(spec=BaseStrategyInterface)
    strategy.name = "test_strategy"
    strategy.strategy_id = "test_001"
    strategy.strategy_type = StrategyType.TREND_FOLLOWING
    strategy.status = StrategyStatus.STOPPED
    strategy.get_current_position = Mock(return_value=None)
    strategy.get_performance_metrics = Mock(return_value={})
    strategy.start = AsyncMock()
    strategy.validate_signal = AsyncMock(return_value=True)
    return strategy


@pytest.fixture(scope="session")
def mock_risk_manager():
    """Mock risk manager."""
    risk_manager = Mock(spec=BaseRiskManager)
    risk_manager.validate_allocation = Mock(return_value=True)
    risk_manager.calculate_position_size = Mock(return_value=Decimal("1000"))
    risk_manager.assess_portfolio_risk = Mock(return_value={"total_risk": 0.15})
    return risk_manager


class TestStrategyAllocation:
    """Test StrategyAllocation class."""
    
    def test_strategy_allocation_initialization(self, mock_strategy):
        """Test StrategyAllocation initialization."""
        allocation = StrategyAllocation(
            strategy=mock_strategy,
            target_weight=0.25,
            current_weight=0.20,
            allocated_capital=Decimal("25000"),
            max_allocation=0.4,
            min_allocation=0.05
        )
        
        assert allocation.strategy == mock_strategy
        assert allocation.target_weight == 0.25
        assert allocation.current_weight == 0.20
        assert allocation.allocated_capital == Decimal("25000")
        assert allocation.max_allocation == 0.4
        assert allocation.min_allocation == 0.05
        assert allocation.cumulative_pnl == Decimal("0")
        assert allocation.trade_count == 0
        assert allocation.win_rate == 0.0
        assert allocation.sharpe_ratio == 0.0

    def test_strategy_allocation_defaults(self, mock_strategy):
        """Test StrategyAllocation default values."""
        allocation = StrategyAllocation(
            strategy=mock_strategy,
            target_weight=0.25,
            current_weight=0.20,
            allocated_capital=Decimal("25000")
        )
        
        assert allocation.max_allocation == 0.4  # Default
        assert allocation.min_allocation == 0.05  # Default


class TestPortfolioAllocator:
    """Test PortfolioAllocator class."""
    
    def test_portfolio_allocator_initialization(self, mock_risk_manager):
        """Test PortfolioAllocator initialization."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager,
            max_strategies=5,
            rebalance_frequency_hours=12,
            min_strategy_allocation=0.10,
            max_strategy_allocation=0.35
        )
        
        assert allocator.total_capital == Decimal("100000")
        assert allocator.risk_manager == mock_risk_manager
        assert allocator.max_strategies == 5
        assert allocator.rebalance_frequency == timedelta(hours=12)
        assert allocator.min_strategy_allocation == 0.10
        assert allocator.max_strategy_allocation == 0.35
        assert len(allocator.allocations) == 0
        assert len(allocator.strategy_queue) == 0
        assert allocator.portfolio_value == Decimal("100000")
        assert allocator.available_capital == Decimal("100000")
        assert allocator.allocated_capital == Decimal("0")

    def test_portfolio_allocator_defaults(self, mock_risk_manager):
        """Test PortfolioAllocator default values."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager
        )
        
        assert allocator.max_strategies == 10  # Default
        assert allocator.rebalance_frequency == timedelta(hours=24)  # Default
        assert allocator.min_strategy_allocation == 0.05  # Default
        assert allocator.max_strategy_allocation == 0.4  # Default

    @pytest.mark.asyncio
    async def test_add_strategy(self, mock_risk_manager, mock_strategy):
        """Test adding a strategy successfully."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager
        )
        
        with patch.object(allocator, 'rebalance_portfolio', new_callable=AsyncMock):
            result = await allocator.add_strategy(mock_strategy, 0.25)
        
        assert result is True
        assert mock_strategy.name in allocator.allocations
        assert len(allocator.allocations) == 1

    @pytest.mark.asyncio
    async def test_add_strategy_exceeds_max_strategies(self, mock_risk_manager):
        """Test adding too many strategies."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager,
            max_strategies=1
        )
        # Set high threshold to bypass correlation check for this test
        allocator.max_correlation_threshold = 1.5
        
        strategy1 = Mock(spec=BaseStrategyInterface)
        strategy1.name = "strategy1"
        strategy1.status = StrategyStatus.STOPPED
        strategy1.strategy_type = StrategyType.MEAN_REVERSION
        strategy1.start = AsyncMock()
        strategy1.validate_signal = AsyncMock(return_value=True)
        
        strategy2 = Mock(spec=BaseStrategyInterface)
        strategy2.name = "strategy2"
        strategy2.status = StrategyStatus.STOPPED
        strategy2.strategy_type = StrategyType.ARBITRAGE
        strategy2.start = AsyncMock()
        strategy2.validate_signal = AsyncMock(return_value=True)
        
        with patch.object(allocator, 'rebalance_portfolio', new_callable=AsyncMock):
            # First strategy should succeed and be allocated
            assert await allocator.add_strategy(strategy1, 0.3) is True
            assert len(allocator.allocations) == 1
            
            # Second strategy should succeed but be queued (max_strategies=1)
            assert await allocator.add_strategy(strategy2, 0.3) is True
            assert len(allocator.allocations) == 1  # Still only one allocated
            assert len(allocator.strategy_queue) == 1  # One in queue

    @pytest.mark.asyncio
    async def test_add_strategy_weight_too_high(self, mock_risk_manager, mock_strategy):
        """Test adding strategy with weight exceeding maximum."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager,
            max_strategy_allocation=0.3
        )
        
        # Should fail because 0.5 > 0.3 (max_strategy_allocation)
        result = await allocator.add_strategy(mock_strategy, 0.5)
        
        assert result is False
        assert len(allocator.allocations) == 0

    @pytest.mark.asyncio
    async def test_add_strategy_weight_too_low(self, mock_risk_manager, mock_strategy):
        """Test adding strategy with weight below minimum."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager,
            min_strategy_allocation=0.1
        )
        
        # Should fail because 0.05 < 0.1 (min_strategy_allocation)
        result = await allocator.add_strategy(mock_strategy, 0.05)
        
        assert result is False
        assert len(allocator.allocations) == 0

    @pytest.mark.asyncio
    async def test_remove_strategy(self, mock_risk_manager, mock_strategy):
        """Test removing a strategy."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager
        )
        
        with patch.object(allocator, 'rebalance_portfolio', new_callable=AsyncMock):
            await allocator.add_strategy(mock_strategy, 0.25)
        
        result = await allocator.remove_strategy(mock_strategy.name)
        
        assert result is True
        assert len(allocator.allocations) == 0

    @pytest.mark.asyncio
    async def test_remove_strategy_not_found(self, mock_risk_manager):
        """Test removing non-existent strategy."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager
        )
        
        result = await allocator.remove_strategy("non_existent_strategy")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_strategy_allocation(self, mock_risk_manager, mock_strategy):
        """Test getting strategy allocation."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager
        )
        
        with patch.object(allocator, 'rebalance_portfolio', new_callable=AsyncMock):
            await allocator.add_strategy(mock_strategy, 0.25)
        
        allocation = allocator.get_strategy_allocation(mock_strategy)
        
        assert allocation is not None
        assert allocation.strategy == mock_strategy
        assert allocation.target_weight == 0.25

    def test_get_strategy_allocation_not_found(self, mock_risk_manager):
        """Test getting allocation for non-existent strategy."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager
        )
        
        mock_strategy = Mock(spec=BaseStrategyInterface)
        allocation = allocator.get_strategy_allocation(mock_strategy)
        
        assert allocation is None

    @pytest.mark.asyncio
    async def test_calculate_optimal_weights(self, mock_risk_manager):
        """Test calculating optimal weights."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager
        )
        
        # Mock the private method to return simple weights
        with patch.object(allocator, '_calculate_optimal_weights', new_callable=AsyncMock) as mock_calc:
            mock_calc.return_value = {"strategy1": 0.6, "strategy2": 0.4}
            
            weights = await allocator.calculate_optimal_weights()
            
            assert isinstance(weights, dict)
            assert weights == {"strategy1": 0.6, "strategy2": 0.4}

    @pytest.mark.asyncio
    async def test_rebalance_portfolio(self, mock_risk_manager, mock_strategy):
        """Test portfolio rebalancing."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager
        )
        
        # Mock the methods called during rebalancing
        with patch.object(allocator, '_update_strategy_metrics', new_callable=AsyncMock), \
             patch.object(allocator, '_calculate_optimal_weights', new_callable=AsyncMock) as mock_calc, \
             patch.object(allocator, '_execute_rebalancing', new_callable=AsyncMock) as mock_exec, \
             patch.object(allocator, '_calculate_portfolio_metrics', new_callable=AsyncMock) as mock_metrics:
            
            mock_calc.return_value = {"test_strategy": 0.5}
            mock_exec.return_value = []
            mock_metrics.return_value = {"total_return": 0.1}
            
            result = await allocator.rebalance_portfolio()
            
            assert isinstance(result, dict)

    def test_update_strategy_performance(self, mock_risk_manager, mock_strategy):
        """Test updating strategy performance metrics."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager
        )
        
        # Add strategy first
        allocation = StrategyAllocation(
            strategy=mock_strategy,
            target_weight=0.3,
            current_weight=0.0,
            allocated_capital=Decimal("0")
        )
        allocator.allocations[mock_strategy.name] = allocation
        
        # Mock performance metrics
        performance_data = {
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "win_rate": 0.65,
            "volatility": 0.12,
            "max_drawdown": 0.08,
            "cumulative_pnl": 5000.0
        }
        
        result = allocator.update_strategy_performance(mock_strategy, performance_data)
        
        assert result is True
        assert allocation.sharpe_ratio == 1.5
        assert allocation.sortino_ratio == 2.0
        assert allocation.win_rate == 0.65
        assert allocation.volatility == 0.12
        assert allocation.max_drawdown == 0.08
        assert allocation.cumulative_pnl == Decimal("5000.0")

    def test_update_strategy_performance_not_found(self, mock_risk_manager):
        """Test updating performance for non-existent strategy."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager
        )
        
        mock_strategy = Mock(spec=BaseStrategyInterface)
        performance_data = {"sharpe_ratio": 1.0}
        
        result = allocator.update_strategy_performance(mock_strategy, performance_data)
        
        assert result is False

    def test_get_allocation_status(self, mock_risk_manager, mock_strategy):
        """Test getting allocation status."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager
        )
        
        # Add a strategy allocation
        allocation = StrategyAllocation(
            strategy=mock_strategy,
            target_weight=0.3,
            current_weight=0.25,
            allocated_capital=Decimal("25000")
        )
        allocator.allocations[mock_strategy.name] = allocation
        
        status = allocator.get_allocation_status()
        
        assert isinstance(status, dict)
        assert "strategy_allocations" in status
        assert "portfolio_summary" in status
        assert mock_strategy.name in status["strategy_allocations"]

    @pytest.mark.asyncio
    async def test_update_market_regime(self, mock_risk_manager):
        """Test updating market regime."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager
        )
        
        await allocator.update_market_regime(MarketRegime.TRENDING_UP)
        
        assert allocator.current_regime == MarketRegime.TRENDING_UP

    @pytest.mark.asyncio
    async def test_should_rebalance(self, mock_risk_manager):
        """Test should_rebalance method."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager,
            rebalance_frequency_hours=24
        )
        
        # Set last_rebalance to old time to trigger rebalancing
        allocator.last_rebalance = datetime.now(timezone.utc) - timedelta(hours=25)
        
        # Should rebalance now (last rebalance was 25 hours ago, frequency is 24 hours)
        should_rebalance = await allocator.should_rebalance()
        assert should_rebalance is True
        
        # Update last_rebalance to recent time
        allocator.last_rebalance = datetime.now(timezone.utc)
        
        # Should not rebalance immediately after
        should_rebalance = await allocator.should_rebalance()
        assert should_rebalance is False

    def test_allocation_error_handling(self, mock_risk_manager):
        """Test allocation error handling."""
        allocator = PortfolioAllocator(
            total_capital=Decimal("100000"),
            risk_manager=mock_risk_manager
        )
        
        # This should work without errors
        assert len(allocator.allocations) == 0
        assert allocator.total_capital == Decimal("100000")
        
        # Test error conditions don't crash
        assert allocator.get_strategy_allocation(None) is None