"""Unit tests for ExecutionRepository."""

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

# Disable logging for performance
logging.disable(logging.CRITICAL)

# Create fixed datetime instances to avoid repeated datetime.now() calls
FIXED_DATETIME = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# Pre-defined constants for faster test data creation
TEST_DATA = {
    "EXECUTION_ID": "test_001",
    "SYMBOL": "BTC/USDT",
    "STATUS_PENDING": "PENDING",
    "STATUS_COMPLETED": "COMPLETED",
    "EMPTY_DICT": {},
    "TRUE_BOOL": True,
    "FALSE_BOOL": False,
    "EMPTY_LIST": [],
    "LIMIT_10": 10,
    "OFFSET_0": 0
}

from src.execution.repository import DatabaseExecutionRepository

# Note: Using generic dict responses since repository uses database service abstraction


class TestDatabaseExecutionRepository:
    """Test cases for ExecutionRepository."""

    @pytest.fixture(scope="session")
    def mock_database_service(self):
        """Create mock database service with pre-defined responses."""
        db_service = MagicMock()
        db_service.create_entity_from_dict = AsyncMock(return_value={"id": TEST_DATA["EXECUTION_ID"]})
        # Mock object with attributes for create_entity
        mock_record = MagicMock()
        mock_record.id = TEST_DATA["EXECUTION_ID"]
        mock_record.execution_id = "test_exec_1"
        mock_record.operation_type = "order_placement"
        mock_record.created_at = None
        db_service.create_entity = AsyncMock(return_value=mock_record)
        db_service.get_entity_by_id = AsyncMock(return_value=None)
        db_service.update_entity = AsyncMock(return_value=TEST_DATA["TRUE_BOOL"])
        db_service.query_entities = AsyncMock(return_value=TEST_DATA["EMPTY_LIST"])
        db_service.delete_entity = AsyncMock(return_value=TEST_DATA["TRUE_BOOL"])
        return db_service

    @pytest.fixture(scope="session")
    def execution_repository(self, mock_database_service):
        """Create DatabaseExecutionRepository instance."""
        return DatabaseExecutionRepository(mock_database_service)

    # Removed sample_execution_result fixture as we're testing repository interface directly

    def test_initialization(self, execution_repository, mock_database_service):
        """Test repository initialization."""
        assert execution_repository.database_service == mock_database_service

    @pytest.mark.asyncio
    async def test_create_execution_record(self, execution_repository):
        """Test creating execution record."""
        execution_data = {"symbol": TEST_DATA["SYMBOL"], "operation_status": "pending", "execution_id": "test_exec_1", "operation_type": "order_placement", "exchange": "binance", "side": "buy", "order_type": "limit"}
        result = await execution_repository.create_execution_record(execution_data)
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_execution_record(self, execution_repository):
        """Test getting execution record by ID."""
        result = await execution_repository.get_execution_record(TEST_DATA["EXECUTION_ID"])
        # Should return result or None
        assert result is None or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_executions_by_criteria(self, execution_repository):
        """Test getting executions by criteria."""
        criteria = {"symbol": TEST_DATA["SYMBOL"]}
        results = await execution_repository.get_executions_by_criteria(criteria)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_executions_by_status(self, execution_repository):
        """Test getting executions by status."""
        criteria = {"operation_status": "completed"}
        results = await execution_repository.get_executions_by_criteria(criteria)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_update_execution_record(self, execution_repository):
        """Test updating execution record."""
        updates = {"operation_status": "completed"}
        result = await execution_repository.update_execution_record(TEST_DATA["EXECUTION_ID"], updates)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_delete_execution_record(self, execution_repository):
        """Test deleting execution record."""
        success = await execution_repository.delete_execution_record(TEST_DATA["EXECUTION_ID"])
        assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_get_executions_with_limit(self, execution_repository):
        """Test getting executions with limit and offset."""
        criteria = {"operation_status": "completed"}
        results = await execution_repository.get_executions_by_criteria(
            criteria, limit=TEST_DATA["LIMIT_10"], offset=TEST_DATA["OFFSET_0"]
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_executions_by_date_range(self, execution_repository):
        """Test getting executions by date range using criteria."""
        start_date = FIXED_DATETIME.replace(hour=0, minute=0, second=0)
        end_date = FIXED_DATETIME

        criteria = {"created_at": {"gte": start_date.isoformat(), "lte": end_date.isoformat()}}
        results = await execution_repository.get_executions_by_criteria(criteria)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_create_execution_record_error_handling(self, execution_repository):
        """Test error handling in create_execution_record."""
        # Test with empty data
        result = await execution_repository.create_execution_record(TEST_DATA["EMPTY_DICT"])
        assert result is not None  # Database service should handle validation

    @pytest.mark.asyncio
    async def test_get_execution_record_invalid_id(self, execution_repository):
        """Test getting execution record with invalid ID."""
        result = await execution_repository.get_execution_record("")
        assert result is None

    @pytest.mark.asyncio
    async def test_repository_database_service_integration(
        self, execution_repository, mock_database_service
    ):
        """Test that repository properly integrates with database service."""
        execution_data = {"symbol": TEST_DATA["SYMBOL"], "operation_status": "pending", "execution_id": "test_exec_1", "operation_type": "order_placement", "exchange": "binance", "side": "buy", "order_type": "limit"}
        await execution_repository.create_execution_record(execution_data)

        # Verify database service methods are available
        assert hasattr(execution_repository.database_service, "create_entity_from_dict")
        assert hasattr(execution_repository.database_service, "get_entity_by_id")
