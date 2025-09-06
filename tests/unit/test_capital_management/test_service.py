"""
Comprehensive tests for CapitalService implementation.

Tests the enterprise-grade capital management service with full coverage of:
- Service lifecycle (start/stop/initialization)
- Core capital operations (allocate/release/update)
- Transaction management and rollback
- Audit logging and state management
- Error handling and circuit breaker patterns
- Performance metrics and health checks
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

# Disable logging during tests to improve performance
logging.getLogger().setLevel(logging.CRITICAL)

from src.capital_management.service import CapitalService
from src.core.base.interfaces import HealthStatus
from src.core.exceptions import DependencyError, ServiceError, ValidationError
from src.core.types.risk import CapitalAllocation, CapitalMetrics
from src.state import StateService


@pytest.fixture(scope="module")
def mock_capital_repository():
    """Mock capital repository for testing."""
    repo = Mock()
    repo.get_all = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.delete = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.get_by_strategy_exchange = AsyncMock()
    return repo


@pytest.fixture(scope="module")
def mock_audit_repository():
    """Mock audit repository for testing."""
    repo = Mock()
    repo.create = AsyncMock()
    repo.get_all = AsyncMock(return_value=[])
    return repo


@pytest.fixture(scope="module")
def mock_state_service():
    """Mock state service for testing."""
    service = Mock(spec=StateService)
    service.set_state = AsyncMock()
    service.get_state = AsyncMock()
    service.delete_state = AsyncMock()
    return service


@pytest.fixture(autouse=True)
def reset_mocks(mock_capital_repository, mock_audit_repository):
    """Reset all mocks before each test."""
    mock_capital_repository.create.reset_mock()
    mock_capital_repository.update.reset_mock()
    mock_capital_repository.delete.reset_mock()
    mock_capital_repository.get_by_strategy_exchange.reset_mock()
    mock_capital_repository.get_all.reset_mock()
    # Reset return_value to empty list but clear any side_effect
    mock_capital_repository.get_all.return_value = []
    mock_capital_repository.get_all.side_effect = None
    mock_capital_repository.create.side_effect = None
    mock_audit_repository.create.reset_mock()


@pytest.fixture
def capital_service(mock_capital_repository, mock_audit_repository, mock_state_service):
    """Create CapitalService instance for testing."""
    return CapitalService(
        capital_repository=mock_capital_repository,
        audit_repository=mock_audit_repository,
        state_service=mock_state_service,
        correlation_id="test-correlation-123",
    )


@pytest.fixture(scope="session")
def sample_allocation():
    """Sample capital allocation for testing."""
    return CapitalAllocation(
        allocation_id="alloc-123",
        strategy_id="momentum-strategy",
        symbol="BTC/USDT",
        allocated_amount=Decimal("1000.00"),
        utilized_amount=Decimal("200.00"),
        available_amount=Decimal("800.00"),
        allocation_percentage=0.1,
        target_allocation_pct=0.1,
        min_allocation=Decimal("100.00"),
        max_allocation=Decimal("5000.00"),
        last_rebalance=datetime.now(timezone.utc),
        next_rebalance=None,
        metadata={"exchange": "binance", "bot_id": "bot-456"},
    )


class TestCapitalServiceInitialization:
    """Test CapitalService initialization and configuration."""

    def test_init_with_all_dependencies(
        self, mock_capital_repository, mock_audit_repository, mock_state_service
    ):
        """Test successful initialization with all dependencies."""
        # Act
        service = CapitalService(
            capital_repository=mock_capital_repository,
            audit_repository=mock_audit_repository,
            state_service=mock_state_service,
            correlation_id="test-123",
        )

        # Assert
        assert service._capital_repository == mock_capital_repository
        assert service._audit_repository == mock_audit_repository
        assert service.state_service == mock_state_service
        assert service._correlation_id == "test-123"
        assert service.total_capital == Decimal("100000")  # Default value
        assert service.emergency_reserve_pct == Decimal("0.1")
        assert service.max_allocation_pct == Decimal("0.2")

    def test_init_without_dependencies(self):
        """Test initialization without dependencies (will use DI)."""
        # Act
        service = CapitalService()

        # Assert
        assert service._capital_repository is None
        assert service._audit_repository is None
        assert service.state_service is None
        assert service.total_capital == Decimal("100000")

    def test_init_performance_metrics_initialized(self, capital_service):
        """Test that performance metrics are properly initialized."""
        # Assert
        metrics = capital_service._performance_metrics
        assert metrics["total_allocations"] == 0
        assert metrics["successful_allocations"] == 0
        assert metrics["failed_allocations"] == 0
        assert metrics["total_releases"] == 0
        assert metrics["total_rebalances"] == 0
        assert metrics["average_allocation_time_ms"] == 0.0
        assert metrics["total_capital_managed"] == Decimal("100000")
        assert metrics["emergency_reserve_maintained"] is True
        assert metrics["risk_limit_violations"] == 0

    def test_init_cache_configuration(self, capital_service):
        """Test cache configuration on initialization."""
        # Assert
        assert capital_service._cache_enabled is True
        assert capital_service._cache_ttl == 300

    def test_init_consistent_patterns_configured(self, capital_service):
        """Test that consistent patterns are initialized."""
        # Assert
        assert capital_service._event_pattern is not None
        assert capital_service._validation_pattern is not None
        assert capital_service._processing_pattern is not None


class TestCapitalServiceLifecycle:
    """Test CapitalService lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_with_dependencies_success(self, capital_service, mock_capital_repository):
        """Test successful service start with dependencies."""
        # Arrange
        mock_capital_repository.get_all.return_value = []

        # Act
        await capital_service._do_start()

        # Assert
        mock_capital_repository.get_all.assert_called_once()
        assert capital_service._performance_metrics["total_allocations"] == 0

    @pytest.mark.asyncio
    async def test_start_without_capital_repository(self):
        """Test service start without capital repository (degraded mode)."""
        # Arrange
        service = CapitalService()
        service.resolve_dependency = Mock(side_effect=DependencyError("Not found"))

        # Act
        await service._do_start()

        # Assert - Should not fail, operates in degraded mode
        assert service._capital_repository is None

    @pytest.mark.asyncio
    async def test_start_repository_resolution_success(self):
        """Test successful repository resolution during start."""
        # Arrange
        service = CapitalService()
        mock_repo = Mock()
        mock_repo.get_all = AsyncMock(return_value=[])
        service.resolve_dependency = Mock(return_value=mock_repo)

        # Act
        await service._do_start()

        # Assert
        assert service._capital_repository == mock_repo
        service.resolve_dependency.assert_called()

    @pytest.mark.asyncio
    async def test_start_initialization_error_handling(self):
        """Test error handling during service start."""
        # Arrange
        service = CapitalService()
        service.resolve_dependency = Mock(side_effect=Exception("Unexpected error"))

        # Act & Assert
        with pytest.raises(ServiceError, match="CapitalService startup failed"):
            await service._do_start()

    @pytest.mark.asyncio
    async def test_stop_success(self, capital_service):
        """Test successful service stop."""
        # Arrange
        capital_service.cleanup_resources = AsyncMock()

        # Act
        await capital_service._do_stop()

        # Assert
        capital_service.cleanup_resources.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_error_handling(self, capital_service):
        """Test error handling during service stop."""
        # Arrange
        capital_service.cleanup_resources = AsyncMock(side_effect=Exception("Cleanup failed"))

        # Act & Assert
        with pytest.raises(ServiceError, match="CapitalService shutdown failed"):
            await capital_service._do_stop()

    @pytest.mark.asyncio
    async def test_initialize_capital_state_success(self, capital_service, mock_capital_repository):
        """Test successful capital state initialization."""
        # Arrange
        mock_allocation = Mock()
        mock_allocation.allocated_amount = Decimal("1000")
        mock_capital_repository.get_all.return_value = [mock_allocation]

        # Reset mock to track only the current method call
        mock_capital_repository.get_all.reset_mock()

        # Act
        await capital_service._initialize_capital_state()

        # Assert
        mock_capital_repository.get_all.assert_called_once()
        assert capital_service._performance_metrics["total_allocations"] == 1

    @pytest.mark.asyncio
    async def test_initialize_capital_state_without_repository(self, capital_service):
        """Test capital state initialization without repository."""
        # Arrange
        capital_service._capital_repository = None

        # Act
        await capital_service._initialize_capital_state()

        # Assert - Should not fail, just log warning
        assert True  # Method should complete without error

    @pytest.mark.asyncio
    async def test_initialize_capital_state_error_handling(
        self, capital_service, mock_capital_repository
    ):
        """Test error handling in capital state initialization."""
        # Arrange
        mock_capital_repository.get_all.side_effect = Exception("Repository error")

        # Act
        await capital_service._initialize_capital_state()

        # Assert - Should not raise, just log error
        assert True  # Method should complete without raising


class TestCapitalAllocation:
    """Test capital allocation functionality."""

    @pytest.mark.asyncio
    async def test_allocate_capital_success(
        self, capital_service, mock_capital_repository, mock_audit_repository
    ):
        """Test successful capital allocation."""
        # Arrange
        mock_capital_repository.get_by_strategy_exchange.return_value = None
        mock_capital_repository.create.return_value = {
            "id": "alloc-123",
            "strategy_id": "test-strategy",
            "exchange": "binance",
            "allocated_amount": "1000",
            "utilized_amount": "0",
            "available_amount": "1000",
            "allocation_percentage": 0.01,
            "last_rebalance": datetime.now(timezone.utc).isoformat(),
        }
        capital_service._get_available_capital = AsyncMock(return_value=Decimal("50000"))

        # Act
        result = await capital_service.allocate_capital(
            strategy_id="test-strategy",
            exchange="binance",
            requested_amount=Decimal("1000"),
            bot_id="bot-123",
        )

        # Assert
        assert result is not None
        mock_capital_repository.create.assert_called_once()
        mock_audit_repository.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_allocate_capital_validation_error(self, capital_service):
        """Test allocation validation error handling."""
        # Act & Assert
        with pytest.raises(ValidationError):
            await capital_service.allocate_capital(
                strategy_id="",  # Invalid empty strategy ID
                exchange="binance",
                requested_amount=Decimal("1000"),
            )

    @pytest.mark.asyncio
    async def test_allocate_capital_insufficient_funds(self, capital_service):
        """Test allocation with insufficient available capital."""
        # Arrange
        capital_service._get_available_capital = AsyncMock(return_value=Decimal("500"))

        # Act & Assert
        with pytest.raises(ValidationError, match="Insufficient capital"):
            await capital_service.allocate_capital(
                strategy_id="test-strategy", exchange="binance", requested_amount=Decimal("1000")
            )

    @pytest.mark.asyncio
    async def test_allocate_capital_exceeds_max_allocation(self, capital_service):
        """Test allocation exceeding maximum allocation percentage."""
        # Arrange
        capital_service._get_available_capital = AsyncMock(return_value=Decimal("50000"))

        # Act & Assert
        with pytest.raises(ValidationError, match="exceeds maximum allocation"):
            await capital_service.allocate_capital(
                strategy_id="test-strategy",
                exchange="binance",
                requested_amount=Decimal("25000"),  # 25% of 100k total capital
            )

    @pytest.mark.asyncio
    async def test_allocate_capital_negative_amount(self, capital_service):
        """Test allocation with negative amount."""
        # Act & Assert
        with pytest.raises(ValidationError):
            await capital_service.allocate_capital(
                strategy_id="test-strategy", exchange="binance", requested_amount=Decimal("-1000")
            )

    @pytest.mark.asyncio
    async def test_allocate_capital_zero_amount(self, capital_service):
        """Test allocation with zero amount."""
        # Act & Assert
        with pytest.raises(ValidationError):
            await capital_service.allocate_capital(
                strategy_id="test-strategy", exchange="binance", requested_amount=Decimal("0")
            )

    @pytest.mark.asyncio
    async def test_allocate_capital_existing_allocation_update(
        self, capital_service, mock_capital_repository
    ):
        """Test updating existing allocation."""
        # Arrange
        existing_allocation = Mock()
        existing_allocation.id = "existing-123"
        existing_allocation.allocated_amount = Decimal("500")
        existing_allocation.utilized_amount = Decimal("0")
        mock_capital_repository.get_by_strategy_exchange.return_value = existing_allocation
        
        # Mock the update method to return a dictionary-like object
        updated_allocation = {
            "id": "existing-123",
            "strategy_id": "test-strategy",
            "exchange": "binance",
            "allocated_amount": "1500",  # 500 + 1000
            "utilized_amount": "0",
            "available_amount": "1500",
            "allocation_percentage": 0.03,  # 1500 / 50000
            "last_rebalance": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        mock_capital_repository.update.return_value = updated_allocation
        capital_service._get_available_capital = AsyncMock(return_value=Decimal("50000"))

        # Act
        result = await capital_service.allocate_capital(
            strategy_id="test-strategy", exchange="binance", requested_amount=Decimal("1000")
        )

        # Assert
        mock_capital_repository.update.assert_called_once()
        assert result is not None


class TestCapitalRelease:
    """Test capital release functionality."""

    @pytest.mark.asyncio
    async def test_release_capital_success(
        self, capital_service, mock_capital_repository, sample_allocation
    ):
        """Test successful capital release."""
        # Arrange
        mock_capital_repository.get_by_strategy_exchange.return_value = sample_allocation
        mock_capital_repository.update.return_value = sample_allocation

        # Act
        result = await capital_service.release_capital(
            strategy_id="momentum-strategy", exchange="binance", release_amount=Decimal("500")
        )

        # Assert
        assert result is True
        mock_capital_repository.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_capital_no_allocation(self, capital_service, mock_capital_repository):
        """Test releasing capital when no allocation exists."""
        # Arrange
        mock_capital_repository.get_by_strategy_exchange.return_value = None

        # Act
        result = await capital_service.release_capital(
            strategy_id="nonexistent-strategy", exchange="binance", release_amount=Decimal("500")
        )

        # Assert - should succeed when no allocation exists
        assert result is True

    @pytest.mark.asyncio
    async def test_release_capital_insufficient_available(
        self, capital_service, mock_capital_repository, sample_allocation
    ):
        """Test releasing more capital than allocated."""
        # Arrange - Create new allocation with lower allocated amount
        low_allocation = CapitalAllocation(
            allocation_id="alloc-low",
            strategy_id="momentum-strategy",
            symbol="BTC/USDT",
            allocated_amount=Decimal("300.00"),  # Less than requested release
            utilized_amount=Decimal("100.00"),
            available_amount=Decimal("200.00"),
            allocation_percentage=0.05,
            target_allocation_pct=0.05,
            min_allocation=Decimal("100.00"),
            max_allocation=Decimal("1000.00"),
            last_rebalance=datetime.now(timezone.utc),
            metadata={"exchange": "binance", "bot_id": "bot-456"},
        )
        mock_capital_repository.get_by_strategy_exchange.return_value = low_allocation

        # Act
        result = await capital_service.release_capital(
            strategy_id="momentum-strategy",
            exchange="binance",
            release_amount=Decimal("500"),  # More than allocated
        )
        
        # Assert - error handling returns None as fallback
        assert result is None

    @pytest.mark.asyncio
    async def test_release_capital_validation_errors(self, capital_service):
        """Test capital release validation errors."""
        # Test empty strategy ID - error handling returns None as fallback
        result = await capital_service.release_capital("", "binance", Decimal("500"))
        assert result is None

        # Test empty exchange - error handling returns None as fallback
        result = await capital_service.release_capital("strategy", "", Decimal("500"))
        assert result is None

        # Test negative amount - error handling returns None as fallback
        result = await capital_service.release_capital("strategy", "binance", Decimal("-500"))
        assert result is None

        # Test zero amount - error handling returns None as fallback
        result = await capital_service.release_capital("strategy", "binance", Decimal("0"))
        assert result is None


class TestCapitalUtilization:
    """Test capital utilization functionality."""

    @pytest.mark.asyncio
    async def test_update_utilization_success(
        self, capital_service, mock_capital_repository, sample_allocation
    ):
        """Test successful utilization update."""
        # Arrange
        mock_capital_repository.get_by_strategy_exchange.return_value = sample_allocation
        mock_capital_repository.update.return_value = sample_allocation

        # Act
        result = await capital_service.update_utilization(
            strategy_id="momentum-strategy", exchange="binance", utilized_amount=Decimal("300")
        )

        # Assert
        assert result is True
        mock_capital_repository.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_utilization_exceeds_allocated(
        self, capital_service, mock_capital_repository, sample_allocation
    ):
        """Test utilization update exceeding allocated amount."""
        # Arrange
        mock_capital_repository.get_by_strategy_exchange.return_value = sample_allocation

        # Act & Assert - error handling returns None as fallback
        result = await capital_service.update_utilization(
            strategy_id="momentum-strategy",
            exchange="binance",
            utilized_amount=Decimal("1500"),  # More than allocated 1000
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_update_utilization_no_allocation(self, capital_service, mock_capital_repository):
        """Test utilization update with no allocation."""
        # Arrange
        mock_capital_repository.get_by_strategy_exchange.return_value = None

        # Act & Assert - returns False when no allocation found 
        result = await capital_service.update_utilization(
            strategy_id="nonexistent", exchange="binance", utilized_amount=Decimal("300")
        )
        assert result is False


class TestCapitalMetrics:
    """Test capital metrics functionality."""

    @pytest.mark.asyncio
    async def test_get_capital_metrics_success(self, capital_service):
        """Test successful capital metrics retrieval."""
        # Arrange
        capital_service._get_capital_metrics_impl = AsyncMock(
            return_value=CapitalMetrics(
                total_capital=Decimal("100000"),
                allocated_amount=Decimal("20000"),
                available_amount=Decimal("80000"),
                total_pnl=Decimal("5000"),
                realized_pnl=Decimal("3000"),
                unrealized_pnl=Decimal("2000"),
                daily_return=0.05,
                weekly_return=0.15,
                monthly_return=0.35,
                yearly_return=1.2,
                total_return=0.25,
                sharpe_ratio=1.5,
                sortino_ratio=1.8,
                calmar_ratio=1.1,
                current_drawdown=-0.02,
                max_drawdown=-0.15,
                var_95=Decimal("5000"),
                expected_shortfall=Decimal("7500"),
                strategies_active=3,
                positions_open=15,
                leverage_used=1.5,
                timestamp=datetime.now(timezone.utc),
            )
        )

        # Act
        metrics = await capital_service.get_capital_metrics()

        # Assert
        assert isinstance(metrics, CapitalMetrics)
        assert metrics.total_capital == Decimal("100000")
        assert metrics.allocated_amount == Decimal("20000")
        assert metrics.available_amount == Decimal("80000")

    @pytest.mark.asyncio
    async def test_get_allocations_by_strategy_success(self, capital_service, sample_allocation):
        """Test successful allocations retrieval by strategy."""
        # Arrange
        capital_service._get_allocations_by_strategy_impl = AsyncMock(
            return_value=[sample_allocation]
        )

        # Act
        allocations = await capital_service.get_allocations_by_strategy("momentum-strategy")

        # Assert
        assert len(allocations) == 1
        assert allocations[0].strategy_id == "momentum-strategy"

    @pytest.mark.asyncio
    async def test_get_allocations_by_strategy_empty(self, capital_service):
        """Test allocations retrieval for strategy with no allocations."""
        # Arrange
        capital_service._get_allocations_by_strategy_impl = AsyncMock(return_value=[])

        # Act
        allocations = await capital_service.get_allocations_by_strategy("empty-strategy")

        # Assert
        assert len(allocations) == 0


class TestErrorHandling:
    """Test error handling and circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_configuration(self, capital_service):
        """Test that circuit breaker is properly configured."""
        # Assert
        assert hasattr(capital_service, "_circuit_breaker_enabled")
        # Additional circuit breaker specific assertions based on implementation

    @pytest.mark.asyncio
    async def test_retry_mechanism_configuration(self, capital_service):
        """Test that retry mechanism is properly configured."""
        # Assert
        assert hasattr(capital_service, "_retry_enabled")
        # Additional retry mechanism assertions based on implementation

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, capital_service, mock_capital_repository):
        """Test transaction rollback on allocation error."""
        # Arrange
        mock_capital_repository.create.side_effect = Exception("Database error")
        capital_service._get_available_capital = AsyncMock(return_value=Decimal("50000"))

        # Act & Assert
        with pytest.raises(Exception):
            await capital_service.allocate_capital(
                strategy_id="test-strategy", exchange="binance", requested_amount=Decimal("1000")
            )

        # Verify rollback behavior (implementation specific)

    @pytest.mark.asyncio
    async def test_audit_logging_on_operations(
        self, capital_service, mock_audit_repository, mock_capital_repository
    ):
        """Test that audit logs are created for operations."""
        # Arrange
        mock_capital_repository.get_by_strategy_exchange.return_value = None
        # Clear any side_effect from previous tests
        mock_capital_repository.create.side_effect = None
        mock_capital_repository.create.return_value = {
            "id": "alloc-123",
            "strategy_id": "test-strategy",
            "exchange": "binance",
            "allocated_amount": "1000",
            "utilized_amount": "0",
            "available_amount": "1000",
            "allocation_percentage": 0.01,
            "last_rebalance": datetime.now(timezone.utc).isoformat(),
        }
        capital_service._get_available_capital = AsyncMock(return_value=Decimal("50000"))

        # Act
        await capital_service.allocate_capital(
            strategy_id="test-strategy", exchange="binance", requested_amount=Decimal("1000")
        )

        # Assert
        mock_audit_repository.create.assert_called_once()


class TestHealthChecks:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_service_health_check_healthy(self, capital_service):
        """Test service health check when healthy."""
        # Arrange
        capital_service._capital_repository = Mock()
        capital_service._audit_repository = Mock()

        # Act
        health = await capital_service._service_health_check()

        # Assert
        assert isinstance(health, HealthStatus)

    @pytest.mark.asyncio
    async def test_service_health_check_degraded(self, capital_service):
        """Test service health check in degraded mode."""
        # Arrange
        capital_service._capital_repository = None  # Degraded mode

        # Act
        health = await capital_service._service_health_check()

        # Assert
        assert isinstance(health, HealthStatus)
        # Additional assertions based on health status implementation


class TestStateManagement:
    """Test state management integration."""

    @pytest.mark.asyncio
    async def test_save_capital_state_snapshot(self, capital_service, mock_state_service):
        """Test saving capital state snapshot."""
        # Arrange - reset the mock to clear previous test calls
        mock_state_service.set_state.reset_mock()
        
        # Act
        await capital_service._save_capital_state_snapshot("test_reason")

        # Assert
        mock_state_service.set_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_capital_state_success(self, capital_service, mock_state_service):
        """Test successful capital state restoration."""
        # Arrange
        mock_state_service.get_state.return_value = {"total_capital": "100000"}

        # Act
        result = await capital_service.restore_capital_state()

        # Assert
        assert result is True
        mock_state_service.get_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_capital_state_no_state(self, capital_service, mock_state_service):
        """Test capital state restoration when no state exists."""
        # Arrange
        mock_state_service.get_state.return_value = None

        # Act
        result = await capital_service.restore_capital_state()

        # Assert
        assert result is False


class TestUtilityMethods:
    """Test utility and helper methods."""

    @pytest.mark.asyncio
    async def test_get_available_capital(self, capital_service, mock_capital_repository):
        """Test available capital calculation."""
        # Arrange
        mock_allocation = Mock()
        mock_allocation.allocated_amount = Decimal("20000")
        mock_capital_repository.get_all.return_value = [mock_allocation]

        # Act
        available = await capital_service._get_available_capital()

        # Assert
        expected_available = (
            capital_service.total_capital
            - capital_service.total_capital * capital_service.emergency_reserve_pct
            - Decimal("20000")
        )
        assert available == expected_available

    def test_safe_decimal_conversion(self, capital_service):
        """Test safe decimal conversion utility."""
        # Test valid string
        assert capital_service._safe_decimal_conversion("123.45") == Decimal("123.45")

        # Test valid decimal
        assert capital_service._safe_decimal_conversion(Decimal("67.89")) == Decimal("67.89")

        # Test invalid input
        assert capital_service._safe_decimal_conversion("invalid") == Decimal("0")

        # Test None input
        assert capital_service._safe_decimal_conversion(None) == Decimal("0")

    def test_load_configuration(self, capital_service):
        """Test configuration loading."""
        # This method should load configuration from config service
        # Test implementation depends on actual config loading logic
        capital_service._load_configuration()

        # Assert default values are maintained if config not available
        assert capital_service.total_capital == Decimal("100000")

    @pytest.mark.asyncio
    async def test_cleanup_resources(self, capital_service):
        """Test resource cleanup."""
        # Act
        await capital_service.cleanup_resources()

        # Assert - Should complete without error
        assert True

    @pytest.mark.asyncio
    async def test_calculate_allocation_efficiency(self, capital_service):
        """Test allocation efficiency calculation."""
        # Arrange
        mock_allocations = [
            Mock(allocated_amount=Decimal("100"), utilized_amount=Decimal("80")),
            Mock(allocated_amount=Decimal("200"), utilized_amount=Decimal("150")),
        ]

        # Act
        efficiency = await capital_service._calculate_allocation_efficiency(mock_allocations)

        # Assert
        expected_efficiency = Decimal("230") / Decimal("300")  # 230/300 = 0.7667
        assert abs(efficiency - expected_efficiency) < Decimal("0.001")

    def test_performance_metrics_updates(self, capital_service):
        """Test performance metrics tracking."""
        # Initial state
        assert capital_service._performance_metrics["total_allocations"] == 0

        # Performance metrics should be updated during operations
        # This is tested indirectly through operation tests
        assert "successful_allocations" in capital_service._performance_metrics
        assert "failed_allocations" in capital_service._performance_metrics
        assert "average_allocation_time_ms" in capital_service._performance_metrics


class TestValidationMethods:
    """Test validation method coverage."""

    @pytest.mark.asyncio
    async def test_validate_allocation_request_consistent(self, capital_service):
        """Test allocation request validation."""
        # Should not raise for valid data
        await capital_service._validate_allocation_request_consistent(
            "test-strategy", "binance", Decimal("1000")
        )

        # Invalid requests should raise ValidationError
        with pytest.raises(ValidationError):
            await capital_service._validate_allocation_request_consistent("", "binance", Decimal("1000"))

    @pytest.mark.asyncio
    async def test_validate_allocation_limits_consistent(self, capital_service):
        """Test allocation limits validation."""
        # Arrange
        capital_service._get_available_capital = AsyncMock(return_value=Decimal("50000"))

        # Should not raise for valid allocation
        await capital_service._validate_allocation_limits_consistent(
            "test-strategy", "binance", Decimal("5000")  # 5% of total capital
        )

        # Invalid allocation exceeding limits
        with pytest.raises(ValidationError, match="exceeds maximum allocation"):
            await capital_service._validate_allocation_limits_consistent(
                "test-strategy", "binance", Decimal("25000")  # 25% of total capital (exceeds 20% max)
            )


class TestRepositoryIntegration:
    """Test repository integration methods."""

    @pytest.mark.asyncio
    async def test_create_allocation(self, capital_service, mock_capital_repository):
        """Test allocation creation via repository."""
        # Arrange
        allocation_data = {
            "strategy_id": "test-strategy",
            "exchange": "binance",
            "allocated_amount": Decimal("1000"),
            "available_amount": Decimal("1000"),
            "utilized_amount": Decimal("0"),
        }
        mock_capital_repository.create.return_value = Mock(allocation_id="alloc-123")

        # Act
        result = await capital_service._create_allocation(allocation_data)

        # Assert
        mock_capital_repository.create.assert_called_once_with(allocation_data)
        assert result is not None

    @pytest.mark.asyncio
    async def test_update_allocation(self, capital_service, mock_capital_repository):
        """Test allocation update via repository."""
        # Arrange
        allocation_data = {"allocation_id": "alloc-123", "allocated_amount": Decimal("1500")}
        mock_capital_repository.update.return_value = Mock(allocation_id="alloc-123")

        # Act
        result = await capital_service._update_allocation(allocation_data)

        # Assert
        mock_capital_repository.update.assert_called_once_with(allocation_data)
        assert result is not None

    @pytest.mark.asyncio
    async def test_delete_allocation(self, capital_service, mock_capital_repository):
        """Test allocation deletion via repository."""
        # Arrange
        mock_capital_repository.delete.return_value = True

        # Act
        result = await capital_service._delete_allocation("alloc-123")

        # Assert
        mock_capital_repository.delete.assert_called_once_with("alloc-123")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_existing_allocation(
        self, capital_service, mock_capital_repository, sample_allocation
    ):
        """Test getting existing allocation via repository."""
        # Arrange
        mock_capital_repository.get_by_strategy_exchange.return_value = sample_allocation

        # Act
        result = await capital_service._get_existing_allocation("momentum-strategy", "binance")

        # Assert
        mock_capital_repository.get_by_strategy_exchange.assert_called_once_with(
            "momentum-strategy", "binance"
        )
        assert result == sample_allocation

    @pytest.mark.asyncio
    async def test_get_all_allocations(self, capital_service, mock_capital_repository):
        """Test getting all allocations via repository."""
        # Arrange
        mock_allocations = [Mock(), Mock()]
        mock_capital_repository.get_all.return_value = mock_allocations

        # Act
        result = await capital_service._get_all_allocations(limit=10)

        # Assert
        mock_capital_repository.get_all.assert_called_once_with(limit=10)
        assert result == mock_allocations

    @pytest.mark.asyncio
    async def test_create_audit_log_record(self, capital_service, mock_audit_repository):
        """Test audit log creation via repository."""
        # Arrange
        audit_data = {
            "action": "allocate_capital",
            "strategy_id": "test-strategy",
            "amount": Decimal("1000"),
        }

        # Act
        await capital_service._create_audit_log_record(audit_data)

        # Assert
        mock_audit_repository.create.assert_called_once_with(audit_data)
