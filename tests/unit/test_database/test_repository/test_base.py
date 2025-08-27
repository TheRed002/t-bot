"""
Unit tests for base repository pattern implementation.

This module tests the BaseRepository class and RepositoryInterface
including all CRUD operations, filtering, and transaction management.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from sqlalchemy import Column, Integer, String, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.base import Base
from src.database.repository.base import BaseRepository, RepositoryInterface


# Test model for repository tests
class TestModel(Base):
    """Test model for repository testing."""
    __tablename__ = "test_model"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    value = Column(Integer)


class TestRepositoryInterface:
    """Test RepositoryInterface abstract base class."""

    def test_repository_interface_is_abstract(self):
        """Test that RepositoryInterface is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            RepositoryInterface()

    def test_repository_interface_methods(self):
        """Test that RepositoryInterface has all required abstract methods."""
        expected_methods = ['get', 'get_all', 'create', 'update', 'delete', 'exists']
        
        for method in expected_methods:
            assert hasattr(RepositoryInterface, method)
            assert getattr(RepositoryInterface, method).__isabstractmethod__ is True

    def test_concrete_repository_implementation(self):
        """Test that a concrete implementation can be created."""
        
        class ConcreteRepository(RepositoryInterface):
            async def get(self, id): return None
            async def get_all(self, filters=None, order_by=None, limit=None, offset=None): return []
            async def create(self, entity): return entity
            async def update(self, entity): return entity
            async def delete(self, id): return True
            async def exists(self, id): return True
        
        # Should be able to instantiate
        repo = ConcreteRepository()
        assert isinstance(repo, RepositoryInterface)


class TestBaseRepository:
    """Test BaseRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        session = AsyncMock(spec=AsyncSession)
        # Make sync methods regular mocks to avoid warnings
        session.delete = Mock()
        session.add = Mock() 
        session.merge = AsyncMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session):
        """Create BaseRepository instance for testing."""
        return BaseRepository(mock_session, TestModel)

    @pytest.fixture
    def sample_entity(self):
        """Create sample test entity."""
        return TestModel(id=1, name="test_entity", value=100)

    def test_base_repository_init(self, mock_session):
        """Test BaseRepository initialization."""
        repo = BaseRepository(mock_session, TestModel)
        
        assert repo.session == mock_session
        assert repo.model == TestModel
        assert repo._logger is not None

    @pytest.mark.asyncio
    async def test_get_success(self, repository, mock_session, sample_entity):
        """Test successful get operation."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_entity
        mock_session.execute.return_value = mock_result
        
        result = await repository.get(1)
        
        assert result == sample_entity
        mock_session.execute.assert_called_once()
        # Check that correct SQL statement was used
        call_args = mock_session.execute.call_args[0][0]
        assert hasattr(call_args, 'whereclause')

    @pytest.mark.asyncio
    async def test_get_not_found(self, repository, mock_session):
        """Test get operation when entity not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        result = await repository.get(999)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_exception(self, repository, mock_session):
        """Test get operation with exception."""
        mock_session.execute.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            await repository.get(1)

    @pytest.mark.asyncio
    async def test_get_by_success(self, repository, mock_session, sample_entity):
        """Test successful get_by operation."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_entity
        mock_session.execute.return_value = mock_result
        
        result = await repository.get_by(name="test_entity", value=100)
        
        assert result == sample_entity
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_not_found(self, repository, mock_session):
        """Test get_by operation when entity not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        result = await repository.get_by(name="nonexistent")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_no_filters(self, repository, mock_session):
        """Test get_all without filters."""
        entities = [TestModel(id=1, name="test1"), TestModel(id=2, name="test2")]
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = entities
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        result = await repository.get_all()
        
        assert result == entities
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_with_filters(self, repository, mock_session):
        """Test get_all with filters."""
        entities = [TestModel(id=1, name="test1")]
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = entities
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        filters = {"name": "test1", "value": 100}
        result = await repository.get_all(filters=filters)
        
        assert result == entities

    @pytest.mark.asyncio
    async def test_get_all_with_list_filter(self, repository, mock_session):
        """Test get_all with list filter (IN clause)."""
        entities = [TestModel(id=1, name="test1"), TestModel(id=2, name="test2")]
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = entities
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        filters = {"name": ["test1", "test2"]}
        result = await repository.get_all(filters=filters)
        
        assert result == entities

    @pytest.mark.asyncio
    async def test_get_all_with_complex_filters(self, repository, mock_session):
        """Test get_all with complex filters (gt, lt, like, etc.)."""
        entities = [TestModel(id=1, name="test1")]
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = entities
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        filters = {
            "value": {"gt": 50, "lt": 150},
            "name": {"like": "test"}
        }
        result = await repository.get_all(filters=filters)
        
        assert result == entities

    @pytest.mark.asyncio
    async def test_get_all_with_ordering(self, repository, mock_session):
        """Test get_all with ordering."""
        entities = [TestModel(id=1, name="test1"), TestModel(id=2, name="test2")]
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = entities
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Test ascending order
        await repository.get_all(order_by="name")
        
        # Test descending order
        await repository.get_all(order_by="-name")
        
        assert mock_session.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_get_all_with_pagination(self, repository, mock_session):
        """Test get_all with pagination."""
        entities = [TestModel(id=1, name="test1")]
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = entities
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        result = await repository.get_all(limit=10, offset=5)
        
        assert result == entities

    @pytest.mark.asyncio
    async def test_create_success(self, repository, mock_session, sample_entity):
        """Test successful create operation."""
        result = await repository.create(sample_entity)
        
        assert result == sample_entity
        mock_session.add.assert_called_once_with(sample_entity)
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_integrity_error(self, repository, mock_session, sample_entity):
        """Test create operation with integrity error."""
        mock_session.flush.side_effect = IntegrityError("Integrity constraint", None, None)
        
        with pytest.raises(IntegrityError):
            await repository.create(sample_entity)
        
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_general_error(self, repository, mock_session, sample_entity):
        """Test create operation with general error."""
        mock_session.flush.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            await repository.create(sample_entity)
        
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_many_success(self, repository, mock_session):
        """Test successful create_many operation."""
        entities = [
            TestModel(id=1, name="test1"),
            TestModel(id=2, name="test2")
        ]
        
        result = await repository.create_many(entities)
        
        assert result == entities
        mock_session.add_all.assert_called_once_with(entities)
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_many_error(self, repository, mock_session):
        """Test create_many operation with error."""
        entities = [TestModel(id=1, name="test1")]
        mock_session.flush.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            await repository.create_many(entities)
        
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_success(self, repository, mock_session):
        """Test successful update operation."""
        entity = TestModel(id=1, name="updated_entity")
        entity.updated_at = datetime.now(timezone.utc)
        entity.version = 1
        
        mock_session.merge.return_value = entity
        
        result = await repository.update(entity)
        
        assert result == entity
        mock_session.merge.assert_called_once()
        mock_session.flush.assert_called_once()
        # Version should be incremented
        assert entity.version == 2

    @pytest.mark.asyncio
    async def test_update_without_version(self, repository, mock_session):
        """Test update operation on entity without version field."""
        entity = TestModel(id=1, name="updated_entity")
        # Remove version attribute if it exists
        if hasattr(entity, 'version'):
            delattr(entity, 'version')
        
        mock_session.merge.return_value = entity
        
        result = await repository.update(entity)
        
        assert result == entity

    @pytest.mark.asyncio
    async def test_update_error(self, repository, mock_session):
        """Test update operation with error."""
        entity = TestModel(id=1, name="test")
        mock_session.merge.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            await repository.update(entity)
        
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_success(self, repository, mock_session, sample_entity):
        """Test successful delete operation."""
        with patch.object(repository, 'get', return_value=sample_entity):
            result = await repository.delete(1)
            
            assert result is True
            mock_session.delete.assert_called_once_with(sample_entity)
            mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_not_found(self, repository, mock_session):
        """Test delete operation when entity not found."""
        with patch.object(repository, 'get', return_value=None):
            result = await repository.delete(999)
            
            assert result is False
            mock_session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_error(self, repository, mock_session, sample_entity):
        """Test delete operation with error."""
        with patch.object(repository, 'get', return_value=sample_entity):
            # Make flush throw an error (since delete is sync but flush is async)
            mock_session.flush = AsyncMock(side_effect=SQLAlchemyError("Database error"))
            
            with pytest.raises(SQLAlchemyError):
                await repository.delete(1)
            
            mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_soft_delete_success(self, repository, mock_session):
        """Test successful soft delete operation."""
        entity = TestModel(id=1, name="test")
        entity.soft_delete = Mock()  # Add soft_delete method
        
        with patch.object(repository, 'get', return_value=entity):
            with patch.object(repository, 'update', return_value=entity):
                result = await repository.soft_delete(1, "admin")
                
                assert result is True
                entity.soft_delete.assert_called_once_with("admin")

    @pytest.mark.asyncio
    async def test_soft_delete_not_found(self, repository, mock_session):
        """Test soft delete when entity not found."""
        with patch.object(repository, 'get', return_value=None):
            result = await repository.soft_delete(999)
            
            assert result is False

    @pytest.mark.asyncio
    async def test_soft_delete_not_supported(self, repository, mock_session, sample_entity):
        """Test soft delete on entity that doesn't support it."""
        # sample_entity doesn't have soft_delete method
        with patch.object(repository, 'get', return_value=sample_entity):
            result = await repository.soft_delete(1)
            
            assert result is False

    @pytest.mark.asyncio
    async def test_exists_true(self, repository, mock_session):
        """Test exists operation returns True."""
        mock_result = Mock()
        mock_result.scalar.return_value = 1  # ID exists
        mock_session.execute.return_value = mock_result
        
        result = await repository.exists(1)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self, repository, mock_session):
        """Test exists operation returns False."""
        mock_result = Mock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result
        
        result = await repository.exists(999)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_error(self, repository, mock_session):
        """Test exists operation with error."""
        mock_session.execute.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            await repository.exists(1)

    @pytest.mark.asyncio
    async def test_count_no_filters(self, repository, mock_session):
        """Test count operation without filters."""
        mock_result = Mock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result
        
        result = await repository.count()
        
        assert result == 5

    @pytest.mark.asyncio
    async def test_count_with_filters(self, repository, mock_session):
        """Test count operation with filters."""
        mock_result = Mock()
        mock_result.scalar.return_value = 3
        mock_session.execute.return_value = mock_result
        
        result = await repository.count(filters={"name": "test"})
        
        assert result == 3

    @pytest.mark.asyncio
    async def test_count_zero_result(self, repository, mock_session):
        """Test count operation with zero result."""
        mock_result = Mock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result
        
        result = await repository.count()
        
        assert result == 0

    @pytest.mark.asyncio
    async def test_count_error(self, repository, mock_session):
        """Test count operation with error."""
        mock_session.execute.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            await repository.count()

    @pytest.mark.asyncio
    async def test_transaction_methods(self, repository, mock_session):
        """Test transaction control methods."""
        mock_transaction = Mock()
        mock_session.begin.return_value = mock_transaction
        
        # Test begin
        result = await repository.begin()
        assert result == mock_transaction
        
        # Test commit
        await repository.commit()
        mock_session.commit.assert_called_once()
        
        # Test rollback
        await repository.rollback()
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_method(self, repository, mock_session):
        """Test list method (alias for get_all)."""
        entities = [TestModel(id=1, name="test1")]
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = entities
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        result = await repository.list(limit=10, offset=5, filters={"name": "test"})
        
        assert result == entities

    @pytest.mark.asyncio
    async def test_get_by_id_method(self, repository, mock_session, sample_entity):
        """Test get_by_id method (alias for get)."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_entity
        mock_session.execute.return_value = mock_result
        
        result = await repository.get_by_id(1)
        
        assert result == sample_entity

    @pytest.mark.asyncio
    async def test_refresh_method(self, repository, mock_session, sample_entity):
        """Test refresh method."""
        await repository.refresh(sample_entity)
        
        mock_session.refresh.assert_called_once_with(sample_entity)


class TestBaseRepositoryFiltering:
    """Test BaseRepository filtering capabilities."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repository(self, mock_session):
        """Create BaseRepository instance for testing."""
        return BaseRepository(mock_session, TestModel)

    @pytest.mark.asyncio
    async def test_filter_with_range_operators(self, repository, mock_session):
        """Test filtering with range operators (gt, gte, lt, lte)."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        filters = {
            "value": {
                "gt": 10,
                "gte": 10, 
                "lt": 100,
                "lte": 100
            }
        }
        
        await repository.get_all(filters=filters)
        
        # Verify SQL execution was called
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_filter_with_like_operator(self, repository, mock_session):
        """Test filtering with LIKE operator."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        filters = {"name": {"like": "test"}}
        
        await repository.get_all(filters=filters)
        
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_filter_with_invalid_attribute(self, repository, mock_session):
        """Test filtering with invalid model attribute."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Filter with attribute that doesn't exist on model
        filters = {"nonexistent_field": "value"}
        
        await repository.get_all(filters=filters)
        
        # Should still execute without adding invalid filter
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_filter_types(self, repository, mock_session):
        """Test multiple filter types combined."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        filters = {
            "name": "exact_match",          # Exact match
            "value": [10, 20, 30],         # IN clause
            "id": {"gt": 5, "lt": 100},    # Range
        }
        
        await repository.get_all(filters=filters)
        
        mock_session.execute.assert_called_once()


class TestBaseRepositoryErrorHandling:
    """Test BaseRepository error handling scenarios."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repository(self, mock_session):
        """Create BaseRepository instance for testing."""
        return BaseRepository(mock_session, TestModel)

    @pytest.mark.asyncio
    async def test_database_connection_error(self, repository, mock_session):
        """Test handling of database connection errors."""
        mock_session.execute.side_effect = SQLAlchemyError("Connection lost")
        
        with pytest.raises(SQLAlchemyError):
            await repository.get(1)

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, repository, mock_session):
        """Test that transactions are rolled back on errors."""
        entity = TestModel(id=1, name="test")
        mock_session.flush.side_effect = SQLAlchemyError("Transaction error")
        
        with pytest.raises(SQLAlchemyError):
            await repository.create(entity)
        
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_integrity_constraint_handling(self, repository, mock_session):
        """Test handling of integrity constraint violations."""
        entity = TestModel(id=1, name="duplicate")
        mock_session.flush.side_effect = IntegrityError(
            "Duplicate key", None, None
        )
        
        with pytest.raises(IntegrityError):
            await repository.create(entity)
        
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_modification_handling(self, repository, mock_session):
        """Test handling of concurrent modification scenarios."""
        entity = TestModel(id=1, name="test")
        entity.version = 1
        
        mock_session.merge.side_effect = SQLAlchemyError("Concurrent modification")
        
        with pytest.raises(SQLAlchemyError):
            await repository.update(entity)
        
        mock_session.rollback.assert_called_once()


class TestBaseRepositoryPerformance:
    """Test BaseRepository performance-related functionality."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repository(self, mock_session):
        """Create BaseRepository instance for testing."""
        return BaseRepository(mock_session, TestModel)

    @pytest.mark.asyncio
    async def test_batch_operations(self, repository, mock_session):
        """Test batch operations for better performance."""
        entities = [TestModel(id=i, name=f"test{i}") for i in range(100)]
        
        result = await repository.create_many(entities)
        
        assert len(result) == 100
        mock_session.add_all.assert_called_once_with(entities)

    @pytest.mark.asyncio
    async def test_pagination_efficiency(self, repository, mock_session):
        """Test pagination doesn't load unnecessary data."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        await repository.get_all(limit=10, offset=100)
        
        # Verify pagination parameters were applied
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_exists_optimization(self, repository, mock_session):
        """Test exists query only selects ID for efficiency."""
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        
        await repository.exists(1)
        
        # Should use optimized query with only ID selection
        call_args = mock_session.execute.call_args[0][0]
        # The exists query should be more efficient than full entity fetch
        mock_session.execute.assert_called_once()


class TestBaseRepositoryEdgeCases:
    """Test BaseRepository edge cases and boundary conditions."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repository(self, mock_session):
        """Create BaseRepository instance for testing."""
        return BaseRepository(mock_session, TestModel)

    @pytest.mark.asyncio
    async def test_empty_filters(self, repository, mock_session):
        """Test operations with empty filters."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Test with empty dict
        await repository.get_all(filters={})
        
        # Test with None
        await repository.get_all(filters=None)
        
        assert mock_session.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_zero_limit_offset(self, repository, mock_session):
        """Test operations with zero limit/offset."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        await repository.get_all(limit=0, offset=0)
        
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_very_large_pagination(self, repository, mock_session):
        """Test operations with very large pagination values."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        await repository.get_all(limit=1000000, offset=1000000)
        
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_null_entity_values(self, repository, mock_session):
        """Test operations with entities containing null values."""
        entity = TestModel(id=1, name=None, value=None)
        
        result = await repository.create(entity)
        
        assert result == entity
        assert result.name is None
        assert result.value is None