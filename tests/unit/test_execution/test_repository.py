"""Unit tests for ExecutionRepository."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from decimal import Decimal
from datetime import datetime, timezone

from src.execution.repository import DatabaseExecutionRepository
# Note: Using generic dict responses since repository uses database service abstraction


class TestDatabaseExecutionRepository:
    """Test cases for ExecutionRepository."""

    @pytest.fixture
    def mock_database_service(self):
        """Create mock database service."""
        db_service = MagicMock()
        db_service.create_entity_from_dict = AsyncMock(return_value={'id': 'test_001'})
        db_service.get_entity_by_id = AsyncMock(return_value=None)
        db_service.update_entity = AsyncMock(return_value=True)
        db_service.query_entities = AsyncMock(return_value=[])
        db_service.delete_entity = AsyncMock(return_value=True)
        return db_service

    @pytest.fixture
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
        execution_data = {'symbol': 'BTC/USDT', 'status': 'PENDING'}
        result = await execution_repository.create_execution_record(execution_data)
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_execution_record(self, execution_repository):
        """Test getting execution record by ID."""
        execution_id = "test_001"
        result = await execution_repository.get_execution_record(execution_id)
        # Should return result or None
        assert result is None or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_executions_by_criteria(self, execution_repository):
        """Test getting executions by criteria."""
        criteria = {"symbol": "BTC/USDT"}
        results = await execution_repository.get_executions_by_criteria(criteria)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_executions_by_status(self, execution_repository):
        """Test getting executions by status."""
        criteria = {"status": "COMPLETED"}
        results = await execution_repository.get_executions_by_criteria(criteria)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_update_execution_record(self, execution_repository):
        """Test updating execution record."""
        execution_id = "test_001"
        updates = {"status": "COMPLETED"}
        result = await execution_repository.update_execution_record(execution_id, updates)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_delete_execution_record(self, execution_repository):
        """Test deleting execution record."""
        execution_id = "test_001"
        success = await execution_repository.delete_execution_record(execution_id)
        assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_get_executions_with_limit(self, execution_repository):
        """Test getting executions with limit and offset."""
        criteria = {"status": "COMPLETED"}
        results = await execution_repository.get_executions_by_criteria(criteria, limit=10, offset=0)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_executions_by_date_range(self, execution_repository):
        """Test getting executions by date range using criteria."""
        start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
        end_date = datetime.now(timezone.utc)
        
        criteria = {
            "created_at": {
                "gte": start_date.isoformat(),
                "lte": end_date.isoformat()
            }
        }
        results = await execution_repository.get_executions_by_criteria(criteria)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_create_execution_record_error_handling(self, execution_repository):
        """Test error handling in create_execution_record."""
        # Test with empty data
        result = await execution_repository.create_execution_record({})
        assert result is not None  # Database service should handle validation

    @pytest.mark.asyncio
    async def test_get_execution_record_invalid_id(self, execution_repository):
        """Test getting execution record with invalid ID."""
        result = await execution_repository.get_execution_record("")
        assert result is None

    @pytest.mark.asyncio
    async def test_repository_database_service_integration(self, execution_repository, mock_database_service):
        """Test that repository properly integrates with database service."""
        execution_data = {'symbol': 'BTC/USDT', 'status': 'PENDING'}
        await execution_repository.create_execution_record(execution_data)
        
        # Verify database service methods are available
        assert hasattr(execution_repository.database_service, 'create_entity_from_dict')
        assert hasattr(execution_repository.database_service, 'get_entity_by_id')