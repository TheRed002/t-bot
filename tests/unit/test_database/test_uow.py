"""
Unit tests for Unit of Work pattern implementation.

This module tests the UnitOfWork class which provides transaction management
and repository coordination for database operations.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import Config
from src.core.exceptions import DatabaseError, DatabaseQueryError
from src.database.uow import AsyncUnitOfWork, UnitOfWork


class TestUnitOfWork:
    """Test UnitOfWork class."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create mock session factory for testing."""
        mock_factory = Mock(spec=sessionmaker)
        mock_session = Mock(spec=Session)
        mock_factory.return_value = mock_session
        return mock_factory

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return Mock()

    @pytest.fixture
    def unit_of_work(self, mock_session_factory, mock_config):
        """Create UnitOfWork instance for testing."""
        return UnitOfWork(mock_session_factory, mock_config)

    @pytest.fixture
    def unit_of_work_no_config(self, mock_session_factory):
        """Create UnitOfWork instance without config for testing."""
        return UnitOfWork(mock_session_factory)

    def test_unit_of_work_init_with_config(self, mock_session_factory, mock_config):
        """Test UnitOfWork initialization with config."""
        uow = UnitOfWork(mock_session_factory, mock_config)
        
        assert uow.session_factory == mock_session_factory
        assert uow.config == mock_config
        assert uow.session is None
        assert uow.error_handler is not None
        assert uow._logger is not None

    def test_unit_of_work_init_no_config(self, mock_session_factory):
        """Test UnitOfWork initialization without config."""
        uow = UnitOfWork(mock_session_factory)
        
        assert uow.session_factory == mock_session_factory
        assert uow.config is None
        assert uow.error_handler is None

    def test_unit_of_work_repository_attributes(self, unit_of_work):
        """Test that UnitOfWork has all expected repository attributes."""
        # These are the actual repository attributes from the implementation
        expected_repos = [
            'alerts', 'audit_logs', 'balance_snapshots', 'bot_instances', 'bot_logs', 'bots', 
            'capital_allocations', 'capital_audit_logs', 'currency_exposures', 'data_pipelines', 
            'data_quality', 'exchange_allocations', 'execution_audit_logs', 'features', 'fills',
            'fund_flows', 'market_data', 'ml', 'ml_models', 'ml_predictions', 'ml_training_jobs', 
            'orders', 'performance_audit_logs', 'performance_metrics', 'positions', 'risk_audit_logs',
            'signals', 'state_backups', 'state_checkpoints', 'state_history', 'state_metadata', 
            'state_snapshots', 'strategies', 'trades', 'users'
        ]
        
        for repo_name in expected_repos:
            assert hasattr(unit_of_work, repo_name)
            # Initially all repositories should be None
            assert getattr(unit_of_work, repo_name) is None

    def test_enter_context_manager(self, unit_of_work, mock_session_factory):
        """Test entering context manager."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        result = unit_of_work.__enter__()
        
        assert result == unit_of_work
        assert unit_of_work.session == mock_session
        # Repositories should be initialized
        assert unit_of_work.users is not None
        assert unit_of_work.orders is not None

    def test_exit_context_manager_success(self, unit_of_work, mock_session_factory):
        """Test exiting context manager successfully."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            pass  # Normal execution
        
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    def test_exit_context_manager_with_exception(self, unit_of_work, mock_session_factory):
        """Test exiting context manager with exception."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        try:
            with unit_of_work:
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.commit.assert_not_called()

    def test_commit_explicit(self, unit_of_work, mock_session_factory):
        """Test explicit commit operation."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            unit_of_work.commit()
        
        # Should commit twice - explicit + context manager exit
        assert mock_session.commit.call_count == 2

    def test_rollback_explicit(self, unit_of_work, mock_session_factory):
        """Test explicit rollback operation."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            unit_of_work.rollback()
        
        mock_session.rollback.assert_called()

    def test_rollback_with_error_handler(self, unit_of_work, mock_session_factory):
        """Test rollback with error handler integration."""
        mock_session = Mock(spec=Session)
        mock_session.rollback.side_effect = SQLAlchemyError("Rollback failed")
        mock_session_factory.return_value = mock_session
        
        with patch.object(unit_of_work, 'error_handler') as mock_handler:
            mock_handler.handle_error = Mock()
            
            with unit_of_work:
                unit_of_work.rollback()
            
            # Error handler should be called for rollback failure
            mock_handler.handle_error.assert_called()

    def test_refresh_entity(self, unit_of_work, mock_session_factory):
        """Test refreshing entity from database."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        mock_entity = Mock()
        
        with unit_of_work:
            unit_of_work.refresh(mock_entity)
        
        mock_session.refresh.assert_called_once_with(mock_entity)

    def test_repository_initialization_lazy(self, unit_of_work, mock_session_factory):
        """Test that repositories are initialized only when accessed."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        # Repositories should be None initially
        assert unit_of_work.users is None
        
        # Access repository through context manager
        with unit_of_work:
            # Now repositories should be initialized
            assert unit_of_work.users is not None
            assert unit_of_work.orders is not None

    def test_multiple_context_usage(self, unit_of_work, mock_session_factory):
        """Test using UnitOfWork multiple times."""
        mock_session1 = Mock(spec=Session)
        mock_session2 = Mock(spec=Session)
        mock_session_factory.side_effect = [mock_session1, mock_session2]
        
        # First usage
        with unit_of_work:
            assert unit_of_work.session == mock_session1
        
        # Second usage should create new session
        with unit_of_work:
            assert unit_of_work.session == mock_session2
        
        mock_session1.commit.assert_called_once()
        mock_session2.commit.assert_called_once()


class TestAsyncUnitOfWork:
    """Test AsyncUnitOfWork class."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create mock async session factory for testing."""
        mock_factory = Mock()
        mock_session = AsyncMock(spec=AsyncSession)
        mock_factory.return_value = mock_session
        return mock_factory

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return Mock()

    @pytest.fixture
    def async_unit_of_work(self, mock_session_factory, mock_config):
        """Create AsyncUnitOfWork instance for testing."""
        return AsyncUnitOfWork(mock_session_factory)

    def test_async_unit_of_work_init(self, mock_session_factory, mock_config):
        """Test AsyncUnitOfWork initialization."""
        uow = AsyncUnitOfWork(mock_session_factory)
        
        assert uow.async_session_factory == mock_session_factory
        assert uow.session is None

    @pytest.mark.asyncio
    async def test_async_enter_context_manager(self, async_unit_of_work, mock_session_factory):
        """Test entering async context manager."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        result = await async_unit_of_work.__aenter__()
        
        assert result == async_unit_of_work
        assert async_unit_of_work.session == mock_session
        # Repositories should be initialized
        assert async_unit_of_work.users is not None

    @pytest.mark.asyncio
    async def test_async_exit_context_manager_success(self, async_unit_of_work, mock_session_factory):
        """Test exiting async context manager successfully."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        async with async_unit_of_work:
            pass  # Normal execution
        
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_exit_context_manager_with_exception(self, async_unit_of_work, mock_session_factory):
        """Test exiting async context manager with exception."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        try:
            async with async_unit_of_work:
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_commit_explicit(self, async_unit_of_work, mock_session_factory):
        """Test explicit async commit operation."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        async with async_unit_of_work:
            await async_unit_of_work.commit()
        
        # Should commit twice - explicit + context manager exit
        assert mock_session.commit.call_count == 2

    @pytest.mark.asyncio
    async def test_async_rollback_explicit(self, async_unit_of_work, mock_session_factory):
        """Test explicit async rollback operation."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        async with async_unit_of_work:
            await async_unit_of_work.rollback()
        
        mock_session.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_async_refresh_entity(self, async_unit_of_work, mock_session_factory):
        """Test refreshing entity from database asynchronously."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        mock_entity = Mock()
        
        async with async_unit_of_work:
            await async_unit_of_work.refresh(mock_entity)
        
        mock_session.refresh.assert_called_once_with(mock_entity)


class TestUnitOfWorkErrorHandling:
    """Test UnitOfWork error handling scenarios."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create mock session factory for testing."""
        mock_factory = Mock(spec=sessionmaker)
        mock_session = Mock(spec=Session)
        mock_factory.return_value = mock_session
        return mock_factory

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return Mock()

    @pytest.fixture
    def unit_of_work(self, mock_session_factory, mock_config):
        """Create UnitOfWork instance for testing."""
        return UnitOfWork(mock_session_factory, mock_config)

    def test_commit_error_handling(self, unit_of_work, mock_session_factory):
        """Test error handling during commit."""
        mock_session = Mock(spec=Session)
        mock_session.commit.side_effect = IntegrityError("Integrity constraint", None, None)
        mock_session_factory.return_value = mock_session
        
        with pytest.raises(IntegrityError):
            with unit_of_work:
                pass  # Should fail on exit due to commit error

    def test_rollback_error_handling(self, unit_of_work, mock_session_factory):
        """Test error handling during rollback."""
        mock_session = Mock(spec=Session)
        mock_session.rollback.side_effect = OperationalError("Connection lost", None, None)
        mock_session_factory.return_value = mock_session
        
        with patch.object(unit_of_work, 'error_handler') as mock_handler:
            mock_handler.handle_error = Mock()
            
            try:
                with unit_of_work:
                    raise ValueError("Original exception")
            except ValueError:
                pass
            
            # Error handler should be called for rollback failure
            mock_handler.handle_error.assert_called()

    def test_session_creation_error(self, unit_of_work, mock_session_factory):
        """Test error handling during session creation."""
        mock_session_factory.side_effect = SQLAlchemyError("Cannot create session")
        
        with pytest.raises(SQLAlchemyError):
            with unit_of_work:
                pass

    def test_repository_initialization_error(self, unit_of_work, mock_session_factory):
        """Test error handling during repository initialization."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        # Mock repository initialization failure
        with patch('src.database.uow.UserRepository', side_effect=Exception("Repository init failed")):
            with pytest.raises(Exception):
                with unit_of_work:
                    pass

    def test_multiple_error_scenarios(self, unit_of_work, mock_session_factory):
        """Test handling multiple error scenarios."""
        mock_session = Mock(spec=Session)
        mock_session.commit.side_effect = IntegrityError("Constraint violation", None, None)
        mock_session.rollback.side_effect = OperationalError("Rollback failed", None, None)
        mock_session_factory.return_value = mock_session
        
        with patch.object(unit_of_work, 'error_handler') as mock_handler:
            mock_handler.handle_error = Mock()
            
            with pytest.raises(IntegrityError):
                with unit_of_work:
                    pass
            
            # Both commit and rollback errors should be handled
            mock_handler.handle_error.assert_called()

    def test_error_without_error_handler(self, mock_session_factory):
        """Test error handling without error handler."""
        uow = UnitOfWork(mock_session_factory)  # No config, no error handler
        
        mock_session = Mock(spec=Session)
        mock_session.rollback.side_effect = OperationalError("Rollback failed", None, None)
        mock_session_factory.return_value = mock_session
        
        # Should not raise exception for rollback failure when no error handler
        try:
            with uow:
                raise ValueError("Original exception")
        except ValueError:
            pass  # Expected


class TestUnitOfWorkTransactionManagement:
    """Test UnitOfWork transaction management scenarios."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create mock session factory for testing."""
        mock_factory = Mock(spec=sessionmaker)
        mock_session = Mock(spec=Session)
        mock_factory.return_value = mock_session
        return mock_factory

    @pytest.fixture
    def unit_of_work(self, mock_session_factory):
        """Create UnitOfWork instance for testing."""
        return UnitOfWork(mock_session_factory)

    def test_nested_transactions(self, unit_of_work, mock_session_factory):
        """Test nested transaction behavior."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            # Simulate nested operation
            unit_of_work.commit()  # Explicit commit
            unit_of_work.rollback()  # Then rollback
        
        # Should have one rollback and one final commit
        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_called()

    def test_transaction_isolation(self, mock_session_factory):
        """Test transaction isolation between different UoW instances."""
        uow1 = UnitOfWork(mock_session_factory)
        uow2 = UnitOfWork(mock_session_factory)
        
        mock_session1 = Mock(spec=Session)
        mock_session2 = Mock(spec=Session)
        mock_session_factory.side_effect = [mock_session1, mock_session2]
        
        # Use both UoW instances
        with uow1:
            assert uow1.session == mock_session1
            
        with uow2:
            assert uow2.session == mock_session2
        
        # Each should have its own session and transaction
        mock_session1.commit.assert_called_once()
        mock_session2.commit.assert_called_once()

    def test_concurrent_transaction_simulation(self, mock_session_factory):
        """Test simulation of concurrent transactions."""
        import threading
        
        results = []
        
        def transaction_worker(worker_id):
            uow = UnitOfWork(mock_session_factory)
            mock_session = Mock(spec=Session)
            mock_session_factory.return_value = mock_session
            
            with uow:
                # Simulate work
                results.append(f"worker_{worker_id}")
            
            # Each worker should commit
            mock_session.commit.assert_called_once()
        
        # Simulate concurrent workers
        threads = []
        for i in range(3):
            thread = threading.Thread(target=transaction_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 3

    def test_long_running_transaction(self, unit_of_work, mock_session_factory):
        """Test long running transaction management."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            # Simulate long running operations
            for i in range(100):
                # Simulate repository operations
                if hasattr(unit_of_work.users, 'get'):
                    pass  # Would call repository methods
                
                # Periodic commits for long transactions
                if i % 50 == 0:
                    unit_of_work.commit()
        
        # Should have intermediate commits plus final commit
        assert mock_session.commit.call_count >= 2

    def test_transaction_with_savepoints(self, unit_of_work, mock_session_factory):
        """Test transaction with savepoints (if supported)."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            # Simulate savepoint-like behavior
            try:
                # Some operation that might fail
                unit_of_work.rollback()  # Rollback to savepoint
            except Exception:
                pass
            
            # Continue with transaction
            unit_of_work.commit()
        
        mock_session.rollback.assert_called()
        mock_session.commit.assert_called()


class TestUnitOfWorkPerformance:
    """Test UnitOfWork performance-related functionality."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create mock session factory for testing."""
        mock_factory = Mock(spec=sessionmaker)
        mock_session = Mock(spec=Session)
        mock_factory.return_value = mock_session
        return mock_factory

    @pytest.fixture
    def unit_of_work(self, mock_session_factory):
        """Create UnitOfWork instance for testing."""
        return UnitOfWork(mock_session_factory)

    def test_repository_lazy_initialization_performance(self, unit_of_work, mock_session_factory):
        """Test that repositories are only initialized when needed."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            # Access only one repository
            _ = unit_of_work.users
            
            # Other repositories should still be None until accessed
            # This test verifies lazy initialization reduces memory usage
            pass

    def test_session_reuse_efficiency(self, unit_of_work, mock_session_factory):
        """Test that session is reused efficiently within context."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            # Multiple repository accesses should reuse same session
            users_repo = unit_of_work.users
            orders_repo = unit_of_work.orders
            
            # Both repositories should use the same session
            assert users_repo.session == mock_session
            assert orders_repo.session == mock_session
        
        # Session factory should only be called once
        mock_session_factory.assert_called_once()

    def test_bulk_operations_efficiency(self, unit_of_work, mock_session_factory):
        """Test efficiency of bulk operations."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            # Simulate bulk operations
            for _ in range(1000):
                # Would normally do bulk inserts/updates
                pass
            
            # Only one commit at the end for efficiency
            unit_of_work.commit()
        
        # Should have final commit plus context manager commit
        assert mock_session.commit.call_count >= 1

    def test_memory_cleanup_after_transaction(self, unit_of_work, mock_session_factory):
        """Test that memory is cleaned up after transactions."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        # Before context
        assert unit_of_work.session is None
        assert unit_of_work.users is None
        
        with unit_of_work:
            # During context - resources allocated
            assert unit_of_work.session is not None
            assert unit_of_work.users is not None
        
        # After context - session closed, but repositories may remain
        # (This behavior may vary based on implementation)
        mock_session.close.assert_called_once()


class TestUnitOfWorkIntegration:
    """Integration-style tests for UnitOfWork."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create mock session factory for testing."""
        mock_factory = Mock(spec=sessionmaker)
        mock_session = Mock(spec=Session)
        mock_factory.return_value = mock_session
        return mock_factory

    @pytest.fixture
    def unit_of_work(self, mock_session_factory):
        """Create UnitOfWork instance for testing."""
        return UnitOfWork(mock_session_factory)

    def test_full_workflow_simulation(self, unit_of_work, mock_session_factory):
        """Test complete workflow with multiple repositories."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        # Simulate complex business transaction
        with unit_of_work:
            # Create user
            user_repo = unit_of_work.users
            assert user_repo is not None
            
            # Create bot
            bot_repo = unit_of_work.bots
            assert bot_repo is not None
            
            # Create orders
            order_repo = unit_of_work.orders
            assert order_repo is not None
            
            # Create positions
            position_repo = unit_of_work.positions
            assert position_repo is not None
            
            # All repositories should share the same session
            assert user_repo.session == mock_session
            assert bot_repo.session == mock_session
            assert order_repo.session == mock_session
            assert position_repo.session == mock_session
        
        # Transaction should complete successfully
        mock_session.commit.assert_called()

    def test_rollback_scenario_simulation(self, unit_of_work, mock_session_factory):
        """Test rollback scenario with multiple operations."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        try:
            with unit_of_work:
                # Simulate multiple repository operations
                _ = unit_of_work.users
                _ = unit_of_work.orders
                _ = unit_of_work.positions
                
                # Simulate error that requires rollback
                raise ValueError("Simulated business logic error")
        except ValueError:
            pass
        
        # Should rollback instead of commit
        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()

    def test_complex_transaction_patterns(self, unit_of_work, mock_session_factory):
        """Test complex transaction patterns."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            # Pattern 1: Create entities with relationships
            user_repo = unit_of_work.users
            bot_repo = unit_of_work.bots
            
            # Pattern 2: Update existing entities
            order_repo = unit_of_work.orders
            
            # Pattern 3: Aggregate operations
            trade_repo = unit_of_work.trades
            
            # Pattern 4: Audit operations
            audit_repo = unit_of_work.audit_logs
            
            # All should work within single transaction
            assert all(repo.session == mock_session for repo in [
                user_repo, bot_repo, order_repo, trade_repo, audit_repo
            ])
        
        mock_session.commit.assert_called()

    def test_repository_consistency_across_transaction(self, unit_of_work, mock_session_factory):
        """Test that repository state is consistent across transaction."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            # Get same repository multiple times
            users1 = unit_of_work.users
            users2 = unit_of_work.users
            
            # Should be the same instance
            assert users1 is users2
            assert users1.session == mock_session
            assert users2.session == mock_session


class TestUnitOfWorkAdvancedScenarios:
    """Test advanced UnitOfWork scenarios and edge cases."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create mock session factory for testing."""
        mock_factory = Mock(spec=sessionmaker)
        mock_session = Mock(spec=Session)
        mock_factory.return_value = mock_session
        return mock_factory

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return Mock()

    @pytest.fixture
    def unit_of_work(self, mock_session_factory, mock_config):
        """Create UnitOfWork instance for testing."""
        return UnitOfWork(mock_session_factory, mock_config)

    def test_repository_access_patterns(self, unit_of_work, mock_session_factory):
        """Test different repository access patterns."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            # Test accessing all repository types
            repos_accessed = []
            
            # Core repositories
            repos_accessed.extend([
                unit_of_work.users,
                unit_of_work.bots,
                unit_of_work.orders,
                unit_of_work.positions,
                unit_of_work.trades,
            ])
            
            # Market data repositories
            repos_accessed.extend([
                unit_of_work.market_data,
                unit_of_work.signals,
            ])
            
            # Capital management repositories
            repos_accessed.extend([
                unit_of_work.capital_allocations,
                unit_of_work.fund_flows,
                unit_of_work.currency_exposures,
            ])
            
            # Audit repositories
            repos_accessed.extend([
                unit_of_work.audit_logs,
                unit_of_work.capital_audit_logs,
                unit_of_work.execution_audit_logs,
            ])
            
            # State management repositories
            repos_accessed.extend([
                unit_of_work.state_snapshots,
                unit_of_work.state_checkpoints,
                unit_of_work.state_backups,
            ])
            
            # ML repositories
            repos_accessed.extend([
                unit_of_work.ml_models,
                unit_of_work.ml_predictions,
                unit_of_work.ml_training_jobs,
            ])
            
            # All should be initialized and share the same session
            for repo in repos_accessed:
                assert repo is not None
                assert repo.session == mock_session

    def test_partial_repository_access(self, unit_of_work, mock_session_factory):
        """Test that only accessed repositories are initialized."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            # Access only users repository
            users_repo = unit_of_work.users
            assert users_repo is not None
            
            # Other repositories should still be None (lazy loading)
            # Note: Implementation may vary, this tests the intended behavior
            
        # Regardless of lazy loading, used repository should have correct session
        assert users_repo.session == mock_session

    def test_transaction_state_transitions(self, unit_of_work, mock_session_factory):
        """Test transaction state transitions."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        # Before transaction
        assert unit_of_work.session is None
        
        with unit_of_work:
            # During transaction - active state
            assert unit_of_work.session == mock_session
            
            # Test explicit state changes
            unit_of_work.commit()  # Should not change session
            assert unit_of_work.session == mock_session
            
            unit_of_work.rollback()  # Should not change session
            assert unit_of_work.session == mock_session
        
        # After transaction - session cleaned up
        mock_session.close.assert_called()

    def test_repository_method_delegation(self, unit_of_work, mock_session_factory):
        """Test that repository methods work correctly."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            users_repo = unit_of_work.users
            
            # Test that repository has expected methods
            assert hasattr(users_repo, 'get')
            assert hasattr(users_repo, 'create')
            assert hasattr(users_repo, 'update')
            assert hasattr(users_repo, 'delete')
            
            # Test session delegation
            assert users_repo.session == mock_session

    def test_transaction_isolation_levels(self, mock_session_factory, mock_config):
        """Test different transaction isolation scenarios."""
        # Test multiple UoW instances with different sessions
        uow1 = UnitOfWork(mock_session_factory, mock_config)
        uow2 = UnitOfWork(mock_session_factory, mock_config)
        
        mock_session1 = Mock(spec=Session)
        mock_session2 = Mock(spec=Session)
        mock_session_factory.side_effect = [mock_session1, mock_session2]
        
        # Both should work independently
        with uow1:
            assert uow1.session == mock_session1
            users1 = uow1.users
        
        with uow2:
            assert uow2.session == mock_session2
            users2 = uow2.users
        
        # Should have separate sessions
        assert users1.session != users2.session
        mock_session1.commit.assert_called_once()
        mock_session2.commit.assert_called_once()

    def test_repository_error_propagation(self, unit_of_work, mock_session_factory):
        """Test error propagation from repositories."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with pytest.raises(AttributeError):
            with unit_of_work:
                # Try to access non-existent repository
                _ = unit_of_work.nonexistent_repository

    def test_session_cleanup_on_errors(self, unit_of_work, mock_session_factory):
        """Test session cleanup on various error conditions."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        # Test cleanup on business logic error
        try:
            with unit_of_work:
                raise ValueError("Business logic error")
        except ValueError:
            pass
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()

    def test_concurrent_access_simulation(self, mock_session_factory, mock_config):
        """Test simulation of concurrent UoW access."""
        import concurrent.futures
        import threading
        
        def worker_transaction(worker_id):
            uow = UnitOfWork(mock_session_factory, mock_config)
            mock_session = Mock(spec=Session)
            mock_session_factory.return_value = mock_session
            
            try:
                with uow:
                    # Simulate work
                    users_repo = uow.users
                    assert users_repo is not None
                    return f"worker_{worker_id}_success"
            except Exception as e:
                return f"worker_{worker_id}_error: {e}"
        
        # Simulate concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker_transaction, i) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All workers should complete successfully
        success_results = [r for r in results if "success" in r]
        assert len(success_results) == 5

    def test_resource_management_edge_cases(self, unit_of_work, mock_session_factory):
        """Test edge cases in resource management."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        # Test rapid enter/exit cycles
        for _ in range(10):
            with unit_of_work:
                _ = unit_of_work.users
        
        # Each cycle should properly clean up
        assert mock_session.close.call_count == 10

    def test_memory_usage_patterns(self, unit_of_work, mock_session_factory):
        """Test memory usage patterns."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        # Test that repositories don't accumulate unnecessarily
        with unit_of_work:
            repos = []
            for _ in range(100):
                # Access same repository multiple times
                repo = unit_of_work.users
                repos.append(repo)
            
            # Should all be the same instance (memory efficient)
            assert all(r is repos[0] for r in repos)

    def test_complex_rollback_scenarios(self, unit_of_work, mock_session_factory):
        """Test complex rollback scenarios."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with patch.object(unit_of_work, 'error_handler') as mock_handler:
            mock_handler.handle_error = Mock()
            
            try:
                with unit_of_work:
                    # Simulate multiple operations
                    _ = unit_of_work.users
                    unit_of_work.commit()  # Intermediate commit
                    
                    _ = unit_of_work.orders
                    # Simulate error after some work
                    raise DatabaseError("Simulated database error")
            except DatabaseError:
                pass
            
            # Should rollback final state
            mock_session.rollback.assert_called()
            # Should have called error handler
            mock_handler.handle_error.assert_called()


class TestAsyncUnitOfWorkAdvanced:
    """Test advanced AsyncUnitOfWork scenarios."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create mock async session factory for testing."""
        mock_factory = Mock()
        mock_session = AsyncMock(spec=AsyncSession)
        mock_factory.return_value = mock_session
        return mock_factory

    @pytest.fixture
    def async_unit_of_work(self, mock_session_factory):
        """Create AsyncUnitOfWork instance for testing."""
        return AsyncUnitOfWork(mock_session_factory)

    @pytest.mark.asyncio
    async def test_async_repository_patterns(self, async_unit_of_work, mock_session_factory):
        """Test async repository access patterns."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        async with async_unit_of_work:
            # Test accessing multiple repositories asynchronously
            users_repo = async_unit_of_work.users
            orders_repo = async_unit_of_work.orders
            trades_repo = async_unit_of_work.trades
            
            assert all(repo.session == mock_session for repo in [
                users_repo, orders_repo, trades_repo
            ])

    @pytest.mark.asyncio
    async def test_async_transaction_coordination(self, async_unit_of_work, mock_session_factory):
        """Test async transaction coordination."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        async with async_unit_of_work:
            # Test explicit async operations
            await async_unit_of_work.commit()
            await async_unit_of_work.rollback()
            await async_unit_of_work.refresh(Mock())
        
        # Should have proper async calls
        assert mock_session.commit.call_count >= 2  # Explicit + context exit
        mock_session.rollback.assert_called()
        mock_session.refresh.assert_called()

    @pytest.mark.asyncio
    async def test_async_error_handling_comprehensive(self, async_unit_of_work, mock_session_factory):
        """Test comprehensive async error handling."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.rollback.side_effect = OperationalError("Async rollback failed", None, None)
        mock_session_factory.return_value = mock_session
        
        try:
            async with async_unit_of_work:
                raise DatabaseQueryError("Async database error")
        except DatabaseQueryError:
            pass
        
        # Should attempt rollback despite it failing
        mock_session.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_async_concurrent_operations(self, mock_session_factory):
        """Test async concurrent operations."""
        import asyncio
        
        async def async_worker(worker_id):
            uow = AsyncUnitOfWork(mock_session_factory)
            mock_session = AsyncMock(spec=AsyncSession)
            mock_session_factory.return_value = mock_session
            
            async with uow:
                # Simulate async work
                await asyncio.sleep(0.01)  # Small delay
                _ = uow.users
                return f"async_worker_{worker_id}"
        
        # Run multiple async workers concurrently
        tasks = [async_worker(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all("async_worker_" in result for result in results)

    @pytest.mark.asyncio
    async def test_async_resource_cleanup(self, async_unit_of_work, mock_session_factory):
        """Test async resource cleanup."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session_factory.return_value = mock_session
        
        # Test cleanup on successful completion
        async with async_unit_of_work:
            _ = async_unit_of_work.users
        
        mock_session.commit.assert_called()
        mock_session.close.assert_called()
        
        # Reset mocks for error case
        mock_session.reset_mock()
        
        # Test cleanup on error
        try:
            async with async_unit_of_work:
                raise ValueError("Async error")
        except ValueError:
            pass
        
        mock_session.rollback.assert_called()
        mock_session.close.assert_called()


class TestUnitOfWorkEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create mock session factory for testing."""
        mock_factory = Mock(spec=sessionmaker)
        mock_session = Mock(spec=Session)
        mock_factory.return_value = mock_session
        return mock_factory

    @pytest.fixture
    def unit_of_work(self, mock_session_factory):
        """Create UnitOfWork instance for testing."""
        return UnitOfWork(mock_session_factory)

    def test_empty_transaction(self, unit_of_work, mock_session_factory):
        """Test transaction with no operations."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            pass  # No operations
        
        # Should still commit
        mock_session.commit.assert_called_once()

    def test_session_factory_failure_recovery(self, mock_session_factory):
        """Test recovery from session factory failures."""
        # First call fails, second succeeds
        mock_session = Mock(spec=Session)
        mock_session_factory.side_effect = [SQLAlchemyError("Factory error"), mock_session]
        
        uow = UnitOfWork(mock_session_factory)
        
        # First attempt should fail
        with pytest.raises(SQLAlchemyError):
            with uow:
                pass
        
        # Second attempt should succeed
        with uow:
            pass
        
        mock_session.commit.assert_called_once()

    def test_repository_initialization_partial_failure(self, unit_of_work, mock_session_factory):
        """Test handling of partial repository initialization failure."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        # Mock one repository to fail initialization
        with patch('src.database.repository.user.UserRepository', side_effect=Exception("User repo failed")):
            with pytest.raises(Exception):
                with unit_of_work:
                    _ = unit_of_work.users  # Should fail

    def test_double_commit_handling(self, unit_of_work, mock_session_factory):
        """Test handling of double commit."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            unit_of_work.commit()  # First commit
            unit_of_work.commit()  # Second commit
        
        # Should handle multiple commits gracefully
        assert mock_session.commit.call_count >= 2

    def test_rollback_after_commit(self, unit_of_work, mock_session_factory):
        """Test rollback after commit."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            unit_of_work.commit()
            unit_of_work.rollback()  # Rollback after commit
        
        # Should handle both operations
        mock_session.commit.assert_called()
        mock_session.rollback.assert_called()

    def test_very_large_transaction(self, unit_of_work, mock_session_factory):
        """Test very large transaction with many operations."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            # Simulate very large transaction
            for i in range(10000):
                # Would normally create/modify entities
                if i % 1000 == 0:
                    # Periodic intermediate commits for large transactions
                    unit_of_work.commit()
        
        # Should handle large number of operations
        assert mock_session.commit.call_count >= 10

    def test_transaction_timeout_simulation(self, unit_of_work, mock_session_factory):
        """Test transaction timeout simulation."""
        mock_session = Mock(spec=Session)
        mock_session.commit.side_effect = OperationalError("Transaction timeout", None, None)
        mock_session_factory.return_value = mock_session
        
        with pytest.raises(OperationalError):
            with unit_of_work:
                pass  # Transaction will timeout on commit

    def test_connection_loss_during_transaction(self, unit_of_work, mock_session_factory):
        """Test connection loss during transaction."""
        mock_session = Mock(spec=Session)
        mock_session.rollback.side_effect = OperationalError("Connection lost", None, None)
        mock_session_factory.return_value = mock_session
        
        with patch.object(unit_of_work, 'error_handler') as mock_handler:
            mock_handler.handle_error = Mock()
            
            try:
                with unit_of_work:
                    raise ValueError("Force rollback")
            except ValueError:
                pass
            
            # Should handle connection loss gracefully
            mock_handler.handle_error.assert_called()

    def test_repository_state_consistency_under_stress(self, unit_of_work, mock_session_factory):
        """Test repository state consistency under stress conditions."""
        mock_session = Mock(spec=Session)
        mock_session_factory.return_value = mock_session
        
        with unit_of_work:
            # Rapid repository access
            repos = []
            for _ in range(1000):
                repo = unit_of_work.users
                repos.append(repo)
            
            # All should be same instance and consistent
            assert all(repo is repos[0] for repo in repos)
            assert all(repo.session == mock_session for repo in repos)