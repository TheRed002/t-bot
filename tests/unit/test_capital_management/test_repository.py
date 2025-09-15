"""
Tests for Capital Management Repository Implementations.

This module tests the service-layer adapter repositories that provide proper
infrastructure abstraction for the CapitalService.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from src.capital_management.repository import AuditRepository, CapitalRepository
from src.core.exceptions import ServiceError
from src.database.models.audit import CapitalAuditLog
from src.database.models.capital import CapitalAllocationDB


class TestCapitalRepository:
    """Test CapitalRepository service-layer adapter."""

    @pytest.fixture
    def mock_capital_allocation_repo(self):
        """Mock the underlying CapitalAllocationRepository."""
        return AsyncMock()

    @pytest.fixture
    def capital_repository(self, mock_capital_allocation_repo):
        """Create CapitalRepository instance with mocked dependencies."""
        return CapitalRepository(mock_capital_allocation_repo)

    @pytest.fixture
    def sample_allocation_data(self):
        """Sample allocation data for testing."""
        return {
            "id": "test-allocation-123",
            "strategy_id": "momentum-strategy",
            "exchange": "binance",
            "allocated_amount": "1000.00",
            "utilized_amount": "500.00",
            "available_amount": "500.00",
            "allocation_percentage": 0.1,
            "last_rebalance": datetime.now(timezone.utc),
        }

    def test_init_requires_capital_allocation_repo(self):
        """Test that initialization requires a valid CapitalAllocationRepository."""
        with pytest.raises(ServiceError, match="CapitalAllocationRepository is required"):
            CapitalRepository(None)

    def test_init_success(self, mock_capital_allocation_repo):
        """Test successful initialization with valid repository."""
        repo = CapitalRepository(mock_capital_allocation_repo)
        assert repo._repo == mock_capital_allocation_repo

    @pytest.mark.asyncio
    async def test_create_success(self, capital_repository, mock_capital_allocation_repo, sample_allocation_data):
        """Test successful allocation creation."""
        # Arrange
        expected_allocation = CapitalAllocationDB(**{
            "id": sample_allocation_data["id"],
            "strategy_id": sample_allocation_data["strategy_id"],
            "exchange": sample_allocation_data["exchange"],
            "allocated_amount": Decimal("1000.00"),
            "utilized_amount": Decimal("500.00"),
            "available_amount": Decimal("500.00"),
            "allocation_percentage": 0.1,
            "last_rebalance": sample_allocation_data["last_rebalance"],
        })
        mock_capital_allocation_repo.create.return_value = expected_allocation

        # Act
        result = await capital_repository.create(sample_allocation_data)

        # Assert
        assert result == expected_allocation
        mock_capital_allocation_repo.create.assert_called_once()
        created_allocation = mock_capital_allocation_repo.create.call_args[0][0]
        assert created_allocation.id == "test-allocation-123"
        assert created_allocation.strategy_id == "momentum-strategy"
        assert created_allocation.exchange == "binance"
        assert created_allocation.allocated_amount == Decimal("1000.00")

    @pytest.mark.asyncio
    async def test_create_handles_repository_error(self, capital_repository, mock_capital_allocation_repo, sample_allocation_data):
        """Test that create properly handles and abstracts repository errors."""
        # Arrange
        mock_capital_allocation_repo.create.side_effect = Exception("Database connection failed")

        # Act & Assert
        with pytest.raises(ServiceError, match="Repository operation failed: Database connection failed"):
            await capital_repository.create(sample_allocation_data)

    @pytest.mark.asyncio
    async def test_update_success(self, capital_repository, mock_capital_allocation_repo, sample_allocation_data):
        """Test successful allocation update."""
        # Arrange
        existing_allocation = CapitalAllocationDB(
            id="test-allocation-123",
            strategy_id="old-strategy",
            exchange="binance",
            allocated_amount=Decimal("800.00"),
            utilized_amount=Decimal("400.00"),
            available_amount=Decimal("400.00"),
            allocation_percentage=0.08,
            last_rebalance=datetime.now(timezone.utc),
        )
        mock_capital_allocation_repo.get.return_value = existing_allocation
        mock_capital_allocation_repo.update.return_value = existing_allocation

        # Act
        result = await capital_repository.update(sample_allocation_data)

        # Assert
        assert result == existing_allocation
        mock_capital_allocation_repo.get.assert_called_once_with("test-allocation-123")
        mock_capital_allocation_repo.update.assert_called_once_with(existing_allocation)
        
        # Verify fields were updated
        assert existing_allocation.strategy_id == "momentum-strategy"
        assert existing_allocation.allocated_amount == Decimal("1000.00")
        assert existing_allocation.utilized_amount == Decimal("500.00")

    @pytest.mark.asyncio
    async def test_update_allocation_not_found(self, capital_repository, mock_capital_allocation_repo, sample_allocation_data):
        """Test update when allocation doesn't exist."""
        # Arrange
        mock_capital_allocation_repo.get.return_value = None

        # Act & Assert
        with pytest.raises(ServiceError, match="Allocation test-allocation-123 not found"):
            await capital_repository.update(sample_allocation_data)

    @pytest.mark.asyncio
    async def test_update_handles_repository_error(self, capital_repository, mock_capital_allocation_repo, sample_allocation_data):
        """Test that update properly handles and abstracts repository errors."""
        # Arrange
        mock_capital_allocation_repo.get.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(ServiceError, match="Repository update operation failed: Database error"):
            await capital_repository.update(sample_allocation_data)

    @pytest.mark.asyncio
    async def test_delete_success(self, capital_repository, mock_capital_allocation_repo):
        """Test successful allocation deletion."""
        # Arrange
        mock_capital_allocation_repo.delete.return_value = None

        # Act
        result = await capital_repository.delete("test-allocation-123")

        # Assert
        assert result is True
        mock_capital_allocation_repo.delete.assert_called_once_with("test-allocation-123")

    @pytest.mark.asyncio
    async def test_delete_handles_repository_error(self, capital_repository, mock_capital_allocation_repo):
        """Test that delete properly handles and abstracts repository errors."""
        # Arrange
        mock_capital_allocation_repo.delete.side_effect = Exception("Delete failed")

        # Act & Assert
        with pytest.raises(ServiceError, match="Repository delete operation failed: Delete failed"):
            await capital_repository.delete("test-allocation-123")

    @pytest.mark.asyncio
    async def test_get_by_strategy_exchange_success(self, capital_repository, mock_capital_allocation_repo):
        """Test successful get by strategy and exchange."""
        # Arrange
        expected_allocation = CapitalAllocationDB(
            id="test-allocation-123",
            strategy_id="momentum-strategy",
            exchange="binance",
            allocated_amount=Decimal("1000.00"),
            utilized_amount=Decimal("500.00"),
            available_amount=Decimal("500.00"),
            allocation_percentage=0.1,
        )
        mock_capital_allocation_repo.find_by_strategy_exchange.return_value = expected_allocation

        # Act
        result = await capital_repository.get_by_strategy_exchange("momentum-strategy", "binance")

        # Assert
        assert result == expected_allocation
        mock_capital_allocation_repo.find_by_strategy_exchange.assert_called_once_with("momentum-strategy", "binance")

    @pytest.mark.asyncio
    async def test_get_by_strategy_exchange_not_found(self, capital_repository, mock_capital_allocation_repo):
        """Test get by strategy and exchange when not found."""
        # Arrange
        mock_capital_allocation_repo.find_by_strategy_exchange.return_value = None

        # Act
        result = await capital_repository.get_by_strategy_exchange("momentum-strategy", "binance")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_strategy_exchange_handles_error(self, capital_repository, mock_capital_allocation_repo):
        """Test that get_by_strategy_exchange properly handles repository errors."""
        # Arrange
        mock_capital_allocation_repo.find_by_strategy_exchange.side_effect = Exception("Query failed")

        # Act & Assert
        with pytest.raises(ServiceError, match="Repository query failed: Query failed"):
            await capital_repository.get_by_strategy_exchange("momentum-strategy", "binance")

    @pytest.mark.asyncio
    async def test_get_by_strategy_success(self, capital_repository, mock_capital_allocation_repo):
        """Test successful get by strategy."""
        # Arrange
        expected_allocations = [
            CapitalAllocationDB(id="alloc-1", strategy_id="momentum-strategy", exchange="binance"),
            CapitalAllocationDB(id="alloc-2", strategy_id="momentum-strategy", exchange="coinbase"),
        ]
        mock_capital_allocation_repo.get_by_strategy.return_value = expected_allocations

        # Act
        result = await capital_repository.get_by_strategy("momentum-strategy")

        # Assert
        assert result == expected_allocations
        mock_capital_allocation_repo.get_by_strategy.assert_called_once_with("momentum-strategy")

    @pytest.mark.asyncio
    async def test_get_by_strategy_handles_error(self, capital_repository, mock_capital_allocation_repo):
        """Test that get_by_strategy properly handles repository errors."""
        # Arrange
        mock_capital_allocation_repo.get_by_strategy.side_effect = Exception("Query failed")

        # Act & Assert
        with pytest.raises(ServiceError, match="Repository query failed: Query failed"):
            await capital_repository.get_by_strategy("momentum-strategy")

    @pytest.mark.asyncio
    async def test_get_all_success(self, capital_repository, mock_capital_allocation_repo):
        """Test successful get all allocations."""
        # Arrange
        expected_allocations = [
            CapitalAllocationDB(id="alloc-1", strategy_id="strategy-1", exchange="binance"),
            CapitalAllocationDB(id="alloc-2", strategy_id="strategy-2", exchange="coinbase"),
        ]
        mock_capital_allocation_repo.get_all.return_value = expected_allocations

        # Act
        result = await capital_repository.get_all()

        # Assert
        assert result == expected_allocations
        mock_capital_allocation_repo.get_all.assert_called_once_with(limit=None)

    @pytest.mark.asyncio
    async def test_get_all_with_limit(self, capital_repository, mock_capital_allocation_repo):
        """Test get all with limit parameter."""
        # Arrange
        expected_allocations = [CapitalAllocationDB(id="alloc-1", strategy_id="strategy-1", exchange="binance")]
        mock_capital_allocation_repo.get_all.return_value = expected_allocations

        # Act
        result = await capital_repository.get_all(limit=10)

        # Assert
        assert result == expected_allocations
        mock_capital_allocation_repo.get_all.assert_called_once_with(limit=10)

    @pytest.mark.asyncio
    async def test_get_all_handles_error(self, capital_repository, mock_capital_allocation_repo):
        """Test that get_all properly handles repository errors."""
        # Arrange
        mock_capital_allocation_repo.get_all.side_effect = Exception("Query failed")

        # Act & Assert
        with pytest.raises(ServiceError, match="Repository query failed: Query failed"):
            await capital_repository.get_all()


class TestAuditRepository:
    """Test AuditRepository service-layer adapter."""

    @pytest.fixture
    def mock_audit_repo(self):
        """Mock the underlying CapitalAuditLogRepository."""
        return AsyncMock()

    @pytest.fixture
    def audit_repository(self, mock_audit_repo):
        """Create AuditRepository instance with mocked dependencies."""
        return AuditRepository(mock_audit_repo)

    @pytest.fixture
    def sample_audit_data(self):
        """Sample audit data for testing."""
        return {
            "id": "audit-123",
            "operation_id": "op-123",
            "operation_type": "allocate",
            "strategy_id": "momentum-strategy",
            "exchange": "binance",
            "bot_id": "bot-001",
            "operation_description": "Capital allocation for momentum strategy",
            "amount": "1000.00",
            "previous_amount": "500.00",
            "new_amount": "1500.00",
            "operation_context": {"reason": "strategy_rebalance"},
            "operation_status": "completed",
            "success": True,
            "error_message": None,
            "authorized_by": "admin@trading.com",
            "requested_at": datetime.now(timezone.utc),
            "executed_at": datetime.now(timezone.utc),
            "source_component": "CapitalService",
            "correlation_id": "corr-123",
        }

    def test_init_requires_audit_repo(self):
        """Test that initialization requires a valid CapitalAuditLogRepository."""
        with pytest.raises(ServiceError, match="CapitalAuditLogRepository is required"):
            AuditRepository(None)

    def test_init_success(self, mock_audit_repo):
        """Test successful initialization with valid audit repository."""
        repo = AuditRepository(mock_audit_repo)
        assert repo._repo == mock_audit_repo

    @pytest.mark.asyncio
    async def test_create_success(self, audit_repository, mock_audit_repo, sample_audit_data):
        """Test successful audit log creation."""
        # Arrange
        expected_audit_log = CapitalAuditLog(**sample_audit_data)
        mock_audit_repo.create.return_value = expected_audit_log

        # Act
        result = await audit_repository.create(sample_audit_data)

        # Assert
        assert result == expected_audit_log
        mock_audit_repo.create.assert_called_once()
        created_log = mock_audit_repo.create.call_args[0][0]
        assert created_log.id == "audit-123"
        assert created_log.operation_id == "op-123"
        assert created_log.operation_type == "allocate"
        assert created_log.strategy_id == "momentum-strategy"
        assert created_log.amount == Decimal("1000.00")

    @pytest.mark.asyncio
    async def test_create_with_minimal_data(self, audit_repository, mock_audit_repo):
        """Test audit log creation with minimal required data."""
        # Arrange
        minimal_data = {
            "id": "audit-123",
            "operation_id": "op-123",
            "operation_type": "allocate",
        }
        expected_audit_log = CapitalAuditLog(**minimal_data)
        mock_audit_repo.create.return_value = expected_audit_log

        # Act
        result = await audit_repository.create(minimal_data)

        # Assert
        assert result == expected_audit_log
        mock_audit_repo.create.assert_called_once()
        created_log = mock_audit_repo.create.call_args[0][0]
        assert created_log.id == "audit-123"
        assert created_log.operation_description == ""
        assert created_log.operation_status == "completed"
        assert created_log.success is True
        assert created_log.source_component == "CapitalService"

    @pytest.mark.asyncio
    async def test_create_with_string_datetime(self, audit_repository, mock_audit_repo):
        """Test audit log creation with string datetime values."""
        # Arrange
        data_with_string_datetime = {
            "id": "audit-123",
            "operation_id": "op-123",
            "operation_type": "allocate",
            "requested_at": "2023-12-01T10:00:00+00:00",
            "executed_at": "2023-12-01T10:05:00+00:00",
        }
        expected_audit_log = CapitalAuditLog(**data_with_string_datetime)
        mock_audit_repo.create.return_value = expected_audit_log

        # Act
        result = await audit_repository.create(data_with_string_datetime)

        # Assert
        assert result == expected_audit_log
        mock_audit_repo.create.assert_called_once()
        created_log = mock_audit_repo.create.call_args[0][0]
        assert isinstance(created_log.requested_at, datetime)
        assert isinstance(created_log.executed_at, datetime)

    @pytest.mark.asyncio
    async def test_create_handles_decimal_conversion(self, audit_repository, mock_audit_repo):
        """Test that create properly converts amounts to Decimal."""
        # Arrange
        data_with_string_amounts = {
            "id": "audit-123",
            "operation_id": "op-123",
            "operation_type": "allocate",
            "amount": "1234.5678",
            "previous_amount": "999.1234",
            "new_amount": "2233.7012",
        }
        expected_audit_log = CapitalAuditLog(**data_with_string_amounts)
        mock_audit_repo.create.return_value = expected_audit_log

        # Act
        result = await audit_repository.create(data_with_string_amounts)

        # Assert
        assert result == expected_audit_log
        created_log = mock_audit_repo.create.call_args[0][0]
        assert created_log.amount == Decimal("1234.5678")
        assert created_log.previous_amount == Decimal("999.1234")
        assert created_log.new_amount == Decimal("2233.7012")

    @pytest.mark.asyncio
    async def test_create_with_none_amounts(self, audit_repository, mock_audit_repo):
        """Test audit log creation with None amount values."""
        # Arrange
        data_with_none_amounts = {
            "id": "audit-123",
            "operation_id": "op-123",
            "operation_type": "allocate",
            "amount": None,
            "previous_amount": None,
            "new_amount": None,
        }
        expected_audit_log = CapitalAuditLog(**data_with_none_amounts)
        mock_audit_repo.create.return_value = expected_audit_log

        # Act
        result = await audit_repository.create(data_with_none_amounts)

        # Assert
        assert result == expected_audit_log
        created_log = mock_audit_repo.create.call_args[0][0]
        assert created_log.amount is None
        assert created_log.previous_amount is None
        assert created_log.new_amount is None

    @pytest.mark.asyncio
    async def test_create_handles_repository_error(self, audit_repository, mock_audit_repo, sample_audit_data):
        """Test that create properly handles and abstracts repository errors."""
        # Arrange
        mock_audit_repo.create.side_effect = Exception("Database write failed")

        # Act & Assert
        with pytest.raises(ServiceError, match="Audit repository operation failed: Database write failed"):
            await audit_repository.create(sample_audit_data)

    @pytest.mark.asyncio
    async def test_create_defaults_timestamps(self, audit_repository, mock_audit_repo):
        """Test that create properly defaults timestamps when not provided."""
        # Arrange
        data_without_timestamps = {
            "id": "audit-123",
            "operation_id": "op-123",
            "operation_type": "allocate",
        }
        expected_audit_log = CapitalAuditLog(**data_without_timestamps)
        mock_audit_repo.create.return_value = expected_audit_log

        # Act
        result = await audit_repository.create(data_without_timestamps)

        # Assert
        assert result == expected_audit_log
        created_log = mock_audit_repo.create.call_args[0][0]
        assert isinstance(created_log.requested_at, datetime)
        assert created_log.executed_at is None  # executed_at should remain None when not provided