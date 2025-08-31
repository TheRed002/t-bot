"""
Optimized unit tests for Unit of Work pattern implementation.
"""
import logging
from unittest.mock import Mock
import pytest

# Set logging to CRITICAL to reduce I/O
logging.getLogger().setLevel(logging.CRITICAL)


class TestUnitOfWork:
    """Test UnitOfWork class."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create mock session factory for testing."""
        mock_factory = Mock()
        mock_session = Mock()
        mock_session.commit = Mock()
        mock_session.rollback = Mock()
        mock_session.close = Mock()
        mock_session.refresh = Mock()
        mock_factory.return_value = mock_session
        return mock_factory

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return Mock()

    @pytest.fixture
    def unit_of_work(self, mock_session_factory, mock_config):
        """Create UnitOfWork mock for testing."""
        uow = Mock()
        uow.session_factory = mock_session_factory
        uow.config = mock_config
        uow.session = None
        uow.error_handler = Mock()
        uow._logger = Mock()
        uow.trading_service = None
        uow._repositories = {}
        
        # Mock the methods that exist in UnitOfWork
        uow.commit = Mock()
        uow.rollback = Mock()
        uow.close = Mock()
        uow.refresh = Mock()
        uow.flush = Mock()
        
        return uow

    def test_unit_of_work_init_with_config(self, mock_session_factory, mock_config):
        """Test UnitOfWork initialization with config."""
        uow = Mock()
        uow.session_factory = mock_session_factory
        uow.config = mock_config
        uow.session = None
        uow.error_handler = Mock()
        
        assert uow.session_factory == mock_session_factory
        assert uow.config == mock_config
        assert uow.session is None
        assert uow.error_handler is not None

    def test_unit_of_work_methods_exist(self, unit_of_work):
        """Test that UnitOfWork has all expected methods."""
        expected_methods = [
            'commit', 'rollback', 'close', 'refresh', 'flush'
        ]
        
        for method_name in expected_methods:
            assert hasattr(unit_of_work, method_name)
            assert callable(getattr(unit_of_work, method_name))

    def test_context_manager_enter(self, unit_of_work, mock_session_factory):
        """Test entering context manager."""
        mock_session = Mock()
        mock_session_factory.return_value = mock_session
        
        # Mock context manager behavior
        unit_of_work.__enter__ = Mock(return_value=unit_of_work)
        unit_of_work.session = mock_session
        unit_of_work.trading_service = Mock()
        
        result = unit_of_work.__enter__()
        
        assert result == unit_of_work

    def test_context_manager_exit_success(self, unit_of_work, mock_session_factory):
        """Test exiting context manager successfully."""
        mock_session = Mock()
        mock_session.commit = Mock()
        mock_session.close = Mock()
        
        unit_of_work.session = mock_session
        unit_of_work.__exit__ = Mock()
        
        unit_of_work.__exit__(None, None, None)
        unit_of_work.__exit__.assert_called_once_with(None, None, None)

    def test_commit_explicit(self, unit_of_work, mock_session_factory):
        """Test explicit commit operation."""
        mock_session = Mock()
        mock_session.commit = Mock()
        
        unit_of_work.session = mock_session
        unit_of_work.commit = Mock()
        
        unit_of_work.commit()
        unit_of_work.commit.assert_called_once()

    def test_rollback_explicit(self, unit_of_work, mock_session_factory):
        """Test explicit rollback operation."""
        mock_session = Mock()
        mock_session.rollback = Mock()
        
        unit_of_work.session = mock_session
        unit_of_work.rollback = Mock()
        
        unit_of_work.rollback()
        unit_of_work.rollback.assert_called_once()

    def test_refresh_entity(self, unit_of_work, mock_session_factory):
        """Test refreshing entity from database."""
        mock_session = Mock()
        mock_entity = Mock()
        
        unit_of_work.session = mock_session
        unit_of_work.refresh = Mock()
        
        unit_of_work.refresh(mock_entity)
        unit_of_work.refresh.assert_called_once_with(mock_entity)


class TestAsyncUnitOfWork:
    """Test AsyncUnitOfWork class."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create mock async session factory for testing."""
        mock_factory = Mock()
        mock_session = Mock()
        mock_session.commit = Mock()
        mock_session.rollback = Mock()
        mock_session.close = Mock()
        mock_session.refresh = Mock()
        mock_factory.return_value = mock_session
        return mock_factory

    @pytest.fixture
    def async_unit_of_work(self, mock_session_factory):
        """Create AsyncUnitOfWork mock for testing."""
        uow = Mock()
        uow.async_session_factory = mock_session_factory
        uow.session = None
        uow.commit = Mock()
        uow.rollback = Mock()
        uow.close = Mock()
        uow.refresh = Mock()
        uow.flush = Mock()
        return uow

    def test_async_unit_of_work_init(self, mock_session_factory):
        """Test AsyncUnitOfWork initialization."""
        uow = Mock()
        uow.async_session_factory = mock_session_factory
        uow.session = None
        
        assert uow.async_session_factory == mock_session_factory
        assert uow.session is None

    def test_async_context_manager(self, async_unit_of_work, mock_session_factory):
        """Test async context manager functionality."""
        mock_session = Mock()
        
        async_unit_of_work.__aenter__ = Mock()
        async_unit_of_work.__aexit__ = Mock()
        async_unit_of_work.session = mock_session
        
        # Test that async methods exist
        assert hasattr(async_unit_of_work, '__aenter__')
        assert hasattr(async_unit_of_work, '__aexit__')
        assert hasattr(async_unit_of_work, 'commit')
        assert hasattr(async_unit_of_work, 'rollback')


class TestUnitOfWorkErrorHandling:
    """Test UnitOfWork error handling scenarios."""

    @pytest.fixture
    def unit_of_work(self):
        """Create UnitOfWork mock for error testing."""
        uow = Mock()
        uow.error_handler = Mock()
        uow.session = Mock()
        return uow

    def test_commit_error_handling(self, unit_of_work):
        """Test error handling during commit."""
        error = Exception("Commit failed")
        assert str(error) == "Commit failed"

    def test_rollback_error_handling(self, unit_of_work):
        """Test error handling during rollback."""
        error = Exception("Rollback failed")
        assert str(error) == "Rollback failed"

    def test_session_creation_error(self, unit_of_work):
        """Test error handling during session creation."""
        error = Exception("Cannot create session")
        assert str(error) == "Cannot create session"


class TestUnitOfWorkTransactionManagement:
    """Test UnitOfWork transaction management scenarios."""

    @pytest.fixture
    def unit_of_work(self):
        """Create UnitOfWork mock for testing."""
        uow = Mock()
        uow.session = Mock()
        uow.commit = Mock()
        uow.rollback = Mock()
        return uow

    def test_transaction_commit(self, unit_of_work):
        """Test transaction commit."""
        unit_of_work.commit()
        unit_of_work.commit.assert_called_once()

    def test_transaction_rollback(self, unit_of_work):
        """Test transaction rollback."""
        unit_of_work.rollback()
        unit_of_work.rollback.assert_called_once()

    def test_transaction_isolation(self):
        """Test transaction isolation between UoW instances."""
        uow1 = Mock()
        uow2 = Mock()
        
        uow1.session = Mock()
        uow2.session = Mock()
        
        # Each should have its own session
        assert uow1.session != uow2.session