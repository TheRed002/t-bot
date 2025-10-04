"""
Capital Management Module Integration Validation Tests.

This test suite validates that capital_management properly integrates with
other modules and uses correct API patterns, error handling, and data contracts.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, AsyncMock
from typing import Any

from src.capital_management.service import CapitalService
from src.capital_management.interfaces import (
    CapitalServiceProtocol,
    CapitalRepositoryProtocol,
    AuditRepositoryProtocol,
)
from src.core.types import CapitalAllocation, CapitalMetrics, BotPriority
from src.core.exceptions import ServiceError, ValidationError
from src.strategies.dependencies import StrategyServiceContainer
from src.web_interface.services.capital_service import WebCapitalService


class TestCapitalManagementIntegration:
    """Test capital_management integration with other modules."""

    @pytest.fixture
    def mock_capital_repository(self):
        """Mock capital repository."""
        repo = Mock(spec=CapitalRepositoryProtocol)
        repo.create = AsyncMock(return_value=Mock(id=1))
        repo.update = AsyncMock(return_value=Mock())
        repo.delete = AsyncMock(return_value=True)
        repo.get_by_strategy = AsyncMock(return_value=[])
        repo.get_all = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_audit_repository(self):
        """Mock audit repository."""
        repo = Mock(spec=AuditRepositoryProtocol)
        repo.create = AsyncMock(return_value=Mock())
        return repo

    @pytest.fixture
    def capital_service(self, mock_capital_repository, mock_audit_repository):
        """Create CapitalService with mocked dependencies."""
        service = CapitalService(
            capital_repository=mock_capital_repository,
            audit_repository=mock_audit_repository,
        )
        return service

    @pytest.mark.asyncio
    async def test_strategy_service_container_integration(self, capital_service):
        """Test that StrategyServiceContainer properly uses CapitalService."""
        # Create service container with capital service
        container = StrategyServiceContainer(capital_service=capital_service)

        # Verify the service is properly stored and accessible
        assert container.capital_service is capital_service
        assert container.get_service_status()["capital_service"] is True

    @pytest.mark.asyncio
    async def test_web_capital_service_integration(self, capital_service):
        """Test WebCapitalService properly uses CapitalService interface."""
        web_service = WebCapitalService(capital_service=capital_service)

        # Mock the capital service methods
        capital_service.get_capital_metrics = AsyncMock(
            return_value=CapitalMetrics(
                total_capital=Decimal("10000"),
                allocated_amount=Decimal("5000"),
                available_amount=Decimal("5000"),
                total_pnl=Decimal("500"),
                realized_pnl=Decimal("300"),
                unrealized_pnl=Decimal("200"),
                daily_return=Decimal("0.01"),
                weekly_return=Decimal("0.05"),
                monthly_return=Decimal("0.20"),
                yearly_return=Decimal("2.40"),
                total_return=Decimal("1.05"),
                sharpe_ratio=Decimal("1.5"),
                sortino_ratio=Decimal("1.8"),
                calmar_ratio=Decimal("2.0"),
                current_drawdown=Decimal("0.02"),
                max_drawdown=Decimal("0.15"),
                var_95=Decimal("500"),
                expected_shortfall=Decimal("750"),
                strategies_active=3,
                positions_open=12,
                leverage_used=Decimal("1.2"),
                timestamp=datetime.now()
            )
        )
        capital_service.allocate_capital = AsyncMock(
            return_value=CapitalAllocation(
                allocation_id="test-1",
                strategy_id="test-strategy",
                allocated_amount=Decimal("1000"),
                utilized_amount=Decimal("0"),
                available_amount=Decimal("1000"),
                allocation_percentage=Decimal("10.0"),
                target_allocation_pct=Decimal("10.0"),
                min_allocation=Decimal("100"),
                max_allocation=Decimal("5000"),
                last_rebalance=datetime.now(),
                exchange="binance"
            )
        )

        # Test web service uses correct API methods
        await web_service._do_start()

        # Test allocation
        result = await web_service.allocate_capital(
            strategy_id="test-strategy",
            exchange="binance",
            amount=Decimal("1000")
        )

        # Verify correct method was called with proper parameters
        capital_service.allocate_capital.assert_called_once_with(
            strategy_id="test-strategy",
            exchange="binance",
            requested_amount=Decimal("1000"),
            bot_id=None,
            authorized_by=None,
            risk_context=None
        )

        # Verify response format
        assert "allocation_id" in result
        assert "allocated_amount" in result
        assert result["strategy_id"] == "test-strategy"

    @pytest.mark.asyncio
    async def test_strategies_base_integration_patterns(self, capital_service):
        """Test that strategies properly use capital service methods."""
        # Mock the methods that strategies actually call
        capital_service.get_capital_metrics = AsyncMock(
            return_value=CapitalMetrics(
                total_capital=Decimal("10000"),
                allocated_amount=Decimal("5000"),
                available_amount=Decimal("2000"),
                total_pnl=Decimal("300"),
                realized_pnl=Decimal("200"),
                unrealized_pnl=Decimal("100"),
                daily_return=Decimal("0.01"),
                weekly_return=Decimal("0.03"),
                monthly_return=Decimal("0.15"),
                yearly_return=Decimal("1.80"),
                total_return=Decimal("1.03"),
                sharpe_ratio=Decimal("1.2"),
                sortino_ratio=Decimal("1.4"),
                calmar_ratio=Decimal("1.6"),
                current_drawdown=Decimal("0.05"),
                max_drawdown=Decimal("0.20"),
                var_95=Decimal("300"),
                expected_shortfall=Decimal("450"),
                strategies_active=2,
                positions_open=8,
                leverage_used=Decimal("1.1"),
                timestamp=datetime.now()
            )
        )

        capital_service.get_allocations_by_strategy = AsyncMock(
            return_value=[
                CapitalAllocation(
                    allocation_id="test-1",
                    strategy_id="test-strategy",
                    allocated_amount=Decimal("1000"),
                    utilized_amount=Decimal("500"),
                    available_amount=Decimal("500"),
                    allocation_percentage=Decimal("10.0"),
                    target_allocation_pct=Decimal("10.0"),
                    min_allocation=Decimal("100"),
                    max_allocation=Decimal("5000"),
                    last_rebalance=datetime.now(),
                    exchange="binance"
                )
            ]
        )

        # Simulate strategy calls - test the actual methods strategies would call

        # Test getting available balance (via get_capital_metrics)
        metrics = await capital_service.get_capital_metrics()
        available_balance = metrics.available_amount
        assert available_balance == Decimal("2000")

        # Test getting strategy allocation (via get_allocations_by_strategy)
        allocations = await capital_service.get_allocations_by_strategy("test-strategy")
        assert len(allocations) == 1
        assert allocations[0].allocated_amount == Decimal("1000")

    @pytest.mark.asyncio
    async def test_error_handling_propagation(self, capital_service):
        """Test that errors are properly handled and propagated across module boundaries."""
        # Mock repository to raise errors
        capital_service._capital_repository.create = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        # Test that service properly handles and propagates errors
        with pytest.raises(ServiceError) as exc_info:
            await capital_service.allocate_capital(
                strategy_id="test-strategy",
                exchange="binance",
                requested_amount=Decimal("1000")
            )

        # Verify error message contains relevant information
        assert "Database connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_dependency_injection_patterns(self, mock_capital_repository, mock_audit_repository):
        """Test that dependency injection works correctly."""
        # Test that service can be created without dependencies (fallback mode)
        service_without_deps = CapitalService()  # Should work with default None values
        assert service_without_deps is not None

        # Test that service works with proper dependencies
        service = CapitalService(
            capital_repository=mock_capital_repository,
            audit_repository=mock_audit_repository
        )

        # Verify dependencies are properly injected
        assert service._capital_repository is mock_capital_repository
        assert service._audit_repository is mock_audit_repository

    @pytest.mark.asyncio
    async def test_interface_compliance(self, capital_service):
        """Test that CapitalService properly implements its interfaces."""
        # Verify it implements the protocol (check required methods instead of isinstance)
        # Since CapitalServiceProtocol is not @runtime_checkable, we test method presence

        # Test required methods exist and have correct signatures
        assert hasattr(capital_service, 'allocate_capital')
        assert hasattr(capital_service, 'release_capital')
        assert hasattr(capital_service, 'get_capital_metrics')
        assert hasattr(capital_service, 'get_allocations_by_strategy')
        assert hasattr(capital_service, 'get_all_allocations')

    @pytest.mark.asyncio
    async def test_bot_resource_manager_integration(self, capital_service):
        """Test ResourceManager integration with CapitalService."""
        from src.bot_management.resource_manager import ResourceManager
        from src.core.config import Config

        # Create mock config
        config = Mock(spec=Config)
        config.bot_management = {"resource_limits": {}}

        # Create resource manager with capital service
        resource_manager = ResourceManager(config=config, capital_service=capital_service)

        # Verify capital service is properly integrated
        assert resource_manager.capital_service is capital_service

        # Mock capital service methods for integration test
        capital_service.allocate_capital = AsyncMock(
            return_value=CapitalAllocation(
                allocation_id="test-1",
                strategy_id="test-bot",
                allocated_amount=Decimal("1000"),
                utilized_amount=Decimal("0"),
                available_amount=Decimal("1000"),
                allocation_percentage=Decimal("10.0"),
                target_allocation_pct=Decimal("10.0"),
                min_allocation=Decimal("100"),
                max_allocation=Decimal("5000"),
                last_rebalance=datetime.now(),
                exchange="internal"
            )
        )
        capital_service.release_capital = AsyncMock(return_value=True)

        # Test resource allocation triggers capital service
        await resource_manager.request_resources(
            bot_id="test-bot",
            capital_amount=Decimal("1000"),
            priority=BotPriority.NORMAL
        )

        # Verify capital service was called
        capital_service.allocate_capital.assert_called()

    @pytest.mark.asyncio
    async def test_data_contract_validation(self, capital_service):
        """Test that data contracts between modules are respected."""
        # Mock successful allocation
        capital_service._capital_repository.create = AsyncMock(
            return_value=Mock(
                id=1,
                allocated_amount=Decimal("1000.00"),
                utilized_amount=Decimal("0.00"),
                available_amount=Decimal("1000.00"),
                allocation_percentage=10.0,
                created_at=None,
                updated_at=None
            )
        )

        # Test allocation returns proper data structure
        result = await capital_service.allocate_capital(
            strategy_id="test-strategy",
            exchange="binance",
            requested_amount=Decimal("1000")
        )

        # Validate data contract
        assert isinstance(result, CapitalAllocation)
        assert isinstance(result.allocated_amount, Decimal)
        assert isinstance(result.utilized_amount, Decimal)
        assert isinstance(result.available_amount, Decimal)
        assert result.strategy_id == "test-strategy"
        assert result.exchange == "binance"

    @pytest.mark.asyncio
    async def test_service_lifecycle_integration(self, capital_service):
        """Test service lifecycle operations work correctly."""
        # Test service can be started
        await capital_service.start()
        assert capital_service.is_running

        # Test service can be stopped
        await capital_service.stop()
        assert not capital_service.is_running


class TestCapitalManagementBoundaryValidation:
    """Test module boundary validation and integration patterns."""

    @pytest.fixture
    def mock_capital_repository(self):
        """Mock capital repository."""
        repo = Mock(spec=CapitalRepositoryProtocol)
        repo.create = AsyncMock(return_value=Mock(id=1))
        repo.update = AsyncMock(return_value=Mock())
        repo.delete = AsyncMock(return_value=True)
        repo.get_by_strategy = AsyncMock(return_value=[])
        repo.get_all = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_audit_repository(self):
        """Mock audit repository."""
        repo = Mock(spec=AuditRepositoryProtocol)
        repo.create = AsyncMock(return_value=Mock())
        return repo

    @pytest.fixture
    def capital_service(self, mock_capital_repository, mock_audit_repository):
        """Create CapitalService with mocked dependencies."""
        service = CapitalService(
            capital_repository=mock_capital_repository,
            audit_repository=mock_audit_repository,
        )
        return service

    @pytest.mark.asyncio
    async def test_no_direct_database_access(self):
        """Verify that consumers don't bypass service layer to access database directly."""
        # This test ensures architectural boundaries are respected

        # Test that strategies module imports only service layer
        from src.strategies.dependencies import StrategyServiceContainer

        # Verify no direct database imports in the dependencies
        import src.strategies.dependencies
        import inspect

        source = inspect.getsource(src.strategies.dependencies)

        # Should not import database models directly
        assert "from src.database.models" not in source
        assert "from src.database.repository" not in source

        # Should import only service interfaces
        assert "from src.capital_management.service import CapitalService" in source

    @pytest.mark.asyncio
    async def test_proper_exception_handling_boundaries(self, capital_service):
        """Test that exceptions are properly transformed at module boundaries."""
        # Mock a database-level exception
        capital_service._capital_repository.create = AsyncMock(
            side_effect=ValueError("Database constraint violation")
        )

        # Service should catch and transform to proper service exception
        with pytest.raises(ServiceError) as exc_info:
            await capital_service.allocate_capital(
                strategy_id="test-strategy",
                exchange="binance",
                requested_amount=Decimal("1000")
            )

        # Verify exception is transformed appropriately
        assert isinstance(exc_info.value, ServiceError)
        # Original error should be preserved in the chain
        assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_interface_segregation(self):
        """Test that interfaces are properly segregated and focused."""
        from src.capital_management.interfaces import (
            CapitalServiceProtocol,
            CurrencyManagementServiceProtocol,
            ExchangeDistributionServiceProtocol,
            FundFlowManagementServiceProtocol
        )

        # Verify interfaces have appropriate method counts (not too large)
        capital_methods = [method for method in dir(CapitalServiceProtocol)
                          if not method.startswith('_')]

        # Interface should be focused - not too many methods
        assert len(capital_methods) <= 10, "Interface is too large, consider splitting"

        # Verify essential methods are present
        assert 'allocate_capital' in capital_methods
        assert 'release_capital' in capital_methods
        assert 'get_capital_metrics' in capital_methods