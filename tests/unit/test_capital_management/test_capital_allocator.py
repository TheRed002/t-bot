"""
Unit tests for CapitalAllocator class.

This module tests the dynamic capital allocation framework including:
- Capital allocation strategies
- Rebalancing logic
- Performance tracking
- Risk-based allocation
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from unittest.mock import Mock, AsyncMock

from src.capital_management.capital_allocator import CapitalAllocator
from src.capital_management.service import CapitalService
from src.core.exceptions import ValidationError, ServiceError
from src.core.types import AllocationStrategy, CapitalAllocation, CapitalMetrics


class TestCapitalAllocator:
    """Test cases for CapitalAllocator class."""

    @pytest.fixture
    def config(self):
        """Create test configuration with capital management settings."""
        config = {
            'total_capital': 100000.0,
            'emergency_reserve_pct': 0.1,
            'allocation_strategy': 'equal_weight',
            'rebalance_frequency_hours': 24,
            'min_allocation_pct': 0.05,
            'max_allocation_pct': 0.4,
            'max_daily_reallocation_pct': 0.1,
            'per_strategy_minimum': {
                'test_strategy': 1000.0
            }
        }
        return config

    @pytest.fixture
    def mock_capital_service(self):
        """Create mock capital service."""
        service = Mock(spec=CapitalService)
        service.allocate_capital = AsyncMock()
        service.release_capital = AsyncMock()
        service.update_utilization = AsyncMock()
        service.get_capital_metrics = AsyncMock()
        service.get_performance_metrics = Mock(return_value={})
        return service

    @pytest.fixture
    def capital_allocator(self, mock_capital_service, config):
        """Create capital allocator instance."""
        allocator = CapitalAllocator(capital_service=mock_capital_service)
        # Manually set config for testing
        allocator.capital_config = config
        allocator.rebalance_frequency_hours = config['rebalance_frequency_hours']
        allocator.max_daily_reallocation_pct = config['max_daily_reallocation_pct']
        return allocator

    def test_initialization(self, capital_allocator, config, mock_capital_service):
        """Test capital allocator initialization."""
        assert capital_allocator.capital_config == config
        assert capital_allocator.capital_service == mock_capital_service
        assert capital_allocator.strategy_performance == {}
        assert capital_allocator.rebalance_frequency_hours == config['rebalance_frequency_hours']
        assert capital_allocator.max_daily_reallocation_pct == config['max_daily_reallocation_pct']
        # Check that last_rebalance is set to a recent time (within last minute)
        assert (datetime.now(timezone.utc) - capital_allocator.last_rebalance).total_seconds() < 60

    @pytest.mark.asyncio
    async def test_allocate_capital_basic(self, capital_allocator, mock_capital_service):
        """Test basic capital allocation."""
        strategy_name = "test_strategy"
        exchange_name = "binance"
        allocation_amount = Decimal("10000")

        # Create mock allocation with expected fields
        from unittest.mock import Mock
        mock_allocation = Mock()
        mock_allocation.strategy_id = strategy_name
        mock_allocation.exchange = exchange_name
        mock_allocation.allocated_amount = allocation_amount
        mock_allocation.allocated_capital = allocation_amount
        mock_allocation.allocation_percentage = 0.1  # Field expected by implementation
        mock_capital_service.allocate_capital.return_value = mock_allocation

        result = await capital_allocator.allocate_capital(
            strategy_name, exchange_name, allocation_amount
        )

        assert result is not None
        assert result.strategy_id == strategy_name
        assert result.allocated_amount == allocation_amount
        
        # Verify service was called correctly
        mock_capital_service.allocate_capital.assert_called_once()

    @pytest.mark.asyncio
    async def test_allocate_capital_insufficient_funds(self, capital_allocator, mock_capital_service):
        """Test allocation with insufficient available capital."""
        strategy_name = "test_strategy"
        exchange_name = "binance"
        allocation_amount = Decimal("100000")  # More than available capital

        # Mock service to raise ValidationError
        mock_capital_service.allocate_capital.side_effect = ValidationError(
            "Insufficient capital"
        )

        with pytest.raises(ValidationError):
            await capital_allocator.allocate_capital(
                strategy_name, exchange_name, allocation_amount
            )

    @pytest.mark.asyncio
    async def test_release_capital(self, capital_allocator, mock_capital_service):
        """Test capital release functionality."""
        strategy_name = "test_strategy"
        exchange_name = "binance"
        release_amount = Decimal("5000")

        # Mock successful release
        mock_capital_service.release_capital.return_value = True

        result = await capital_allocator.release_capital(
            strategy_name, exchange_name, release_amount
        )

        assert result is True
        mock_capital_service.release_capital.assert_called_once_with(
            strategy_id=strategy_name,
            exchange=exchange_name,
            release_amount=release_amount,
            bot_id=None,
            authorized_by="CapitalAllocator"
        )

    @pytest.mark.asyncio
    async def test_get_capital_metrics(self, capital_allocator, mock_capital_service):
        """Test getting capital metrics."""
        # Create mock metrics with expected fields
        from unittest.mock import Mock
        mock_metrics = Mock()
        mock_metrics.total_capital = Decimal("100000")
        mock_metrics.allocated_capital = Decimal("45000")
        mock_metrics.available_capital = Decimal("45000")
        mock_metrics.utilization_rate = 0.45  # Field expected by implementation
        mock_metrics.allocation_efficiency = 0.85  # Field expected by implementation
        mock_capital_service.get_capital_metrics.return_value = mock_metrics

        result = await capital_allocator.get_capital_metrics()

        assert result is not None
        assert result.total_capital == Decimal("100000")
        assert result.allocated_capital == Decimal("45000")
        mock_capital_service.get_capital_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_rebalance_allocations(self, capital_allocator, mock_capital_service):
        """Test rebalancing allocations."""
        # Create mock current metrics indicating rebalance is needed
        from unittest.mock import Mock
        mock_metrics = Mock()
        mock_metrics.total_capital = Decimal("100000")
        mock_metrics.allocated_capital = Decimal("45000")
        mock_metrics.available_capital = Decimal("45000")
        mock_metrics.utilization_rate = 0.3  # Low utilization
        mock_metrics.allocation_efficiency = 0.2  # Low efficiency  
        mock_metrics.allocation_count = 3
        mock_capital_service.get_capital_metrics.return_value = mock_metrics

        # Force last rebalance to be old enough
        capital_allocator.last_rebalance = datetime.now(timezone.utc) - timedelta(hours=25)

        result = await capital_allocator.rebalance_allocations()

        assert isinstance(result, dict)
        mock_capital_service.get_capital_metrics.assert_called()

    @pytest.mark.asyncio
    async def test_update_utilization(self, capital_allocator, mock_capital_service):
        """Test updating utilization."""
        strategy_name = "test_strategy"
        exchange_name = "binance"
        utilized_amount = Decimal("7500")

        # Mock successful update
        mock_capital_service.update_utilization.return_value = True

        await capital_allocator.update_utilization(
            strategy_name, exchange_name, utilized_amount
        )

        mock_capital_service.update_utilization.assert_called_once_with(
            strategy_id=strategy_name,
            exchange=exchange_name,
            utilized_amount=utilized_amount,
            bot_id=None
        )

    @pytest.mark.asyncio
    async def test_get_emergency_reserve(self, capital_allocator, mock_capital_service):
        """Test getting emergency reserve."""
        # Create mock metrics with emergency reserve
        from unittest.mock import Mock
        mock_metrics = Mock()
        mock_metrics.emergency_reserve = Decimal("10000")  # Field expected by implementation
        mock_capital_service.get_capital_metrics.return_value = mock_metrics

        result = await capital_allocator.get_emergency_reserve()

        assert result == Decimal("10000")
        mock_capital_service.get_capital_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_allocation_summary(self, capital_allocator, mock_capital_service):
        """Test getting allocation summary."""
        # Create mock metrics
        from unittest.mock import Mock
        mock_metrics = Mock()
        mock_metrics.total_capital = Decimal("100000")
        mock_metrics.allocated_capital = Decimal("45000")
        mock_metrics.available_capital = Decimal("45000")
        mock_metrics.allocation_count = 3
        mock_metrics.emergency_reserve = Decimal("10000")
        mock_metrics.utilization_rate = 0.45
        mock_metrics.allocation_efficiency = 0.85
        mock_capital_service.get_capital_metrics.return_value = mock_metrics
        
        # Mock performance metrics
        mock_capital_service.get_performance_metrics.return_value = {
            "successful_allocations": 10,
            "failed_allocations": 2,
            "total_releases": 5,
            "average_allocation_time_ms": 250.0
        }

        result = await capital_allocator.get_allocation_summary()

        assert isinstance(result, dict)
        assert result["total_allocations"] == 3
        assert result["total_allocated"] == Decimal("45000")
        assert result["total_capital"] == Decimal("100000")
        assert "service_metrics" in result

    @pytest.mark.asyncio
    async def test_service_error_handling(self, capital_allocator, mock_capital_service):
        """Test handling of service errors."""
        # Mock service to raise ServiceError
        mock_capital_service.allocate_capital.side_effect = ServiceError("Service unavailable")

        with pytest.raises(ServiceError):
            await capital_allocator.allocate_capital("test", "binance", Decimal("1000"))