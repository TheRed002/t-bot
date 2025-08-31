"""
Functional tests for repository factory.

Tests that actually exercise the repository factory code to ensure coverage.
"""

import pytest
from unittest.mock import Mock, MagicMock

# Import modules to ensure they are loaded for coverage
from src.database.repository_factory import RepositoryFactory


class MockRepository:
    """Mock repository for functional testing."""
    
    def __init__(self, session):
        self.session = session
        self.initialized = True


class TestRepositoryFactoryFunctional:
    """Functional tests that exercise repository factory code."""

    def test_repository_factory_instantiation(self):
        """Test that RepositoryFactory can be instantiated."""
        # This will trigger the __init__ method for coverage
        factory = RepositoryFactory()
        
        assert factory is not None
        assert factory._dependency_injector is None
        assert isinstance(factory._registered_repositories, dict)
        assert len(factory._registered_repositories) == 0

    def test_repository_factory_with_dependency_injector(self):
        """Test instantiation with dependency injector."""
        mock_di = Mock()
        factory = RepositoryFactory(dependency_injector=mock_di)
        
        assert factory._dependency_injector is mock_di

    def test_create_repository_direct_instantiation(self):
        """Test creating repository through direct instantiation path."""
        factory = RepositoryFactory()
        mock_session = Mock()
        
        # This exercises the main create_repository method
        repository = factory.create_repository(MockRepository, mock_session)
        
        assert isinstance(repository, MockRepository)
        assert repository.session is mock_session
        assert repository.initialized is True

    def test_create_repository_with_dependency_injection_has_method(self):
        """Test creating repository with DI that has 'has' method."""
        mock_di = Mock()
        mock_di.has = Mock(return_value=True)
        
        mock_repository = MockRepository(Mock())
        mock_di.resolve = Mock(return_value=mock_repository)
        
        factory = RepositoryFactory(dependency_injector=mock_di)
        mock_session = Mock()
        
        # This exercises the DI path with has() method
        result = factory.create_repository(MockRepository, mock_session)
        
        assert result is mock_repository
        # DI interface doesn't have 'has' method in actual implementation
        mock_di.resolve.assert_called_once_with("MockRepository")

    def test_create_repository_with_dependency_injection_container_has(self):
        """Test creating repository with DI container that has 'has' method."""
        mock_di = Mock()
        mock_di.has = Mock(return_value=False)
        mock_di._container = Mock()
        mock_di._container.has = Mock(return_value=True)
        
        mock_repository = MockRepository(Mock())
        mock_di.resolve = Mock(return_value=mock_repository)
        
        factory = RepositoryFactory(dependency_injector=mock_di)
        mock_session = Mock()
        
        # This exercises the container DI path
        result = factory.create_repository(MockRepository, mock_session)
        
        assert result is mock_repository

    def test_create_repository_di_fallback_path(self):
        """Test repository creation fallback when DI fails."""
        mock_di = Mock()
        mock_di.has = Mock(return_value=True)
        mock_di.resolve = Mock(side_effect=Exception("DI resolution failed"))
        
        factory = RepositoryFactory(dependency_injector=mock_di)
        mock_session = Mock()
        
        # Should raise RuntimeError as per actual implementation
        with pytest.raises(RuntimeError, match="Repository creation failed"):
            factory.create_repository(MockRepository, mock_session)

    def test_create_repository_complete_failure(self):
        """Test repository creation when all attempts fail."""
        class FailingRepository:
            def __init__(self, session):
                raise ValueError("Repository initialization failed")
        
        factory = RepositoryFactory()
        mock_session = Mock()
        
        # This exercises the error handling path
        with pytest.raises(RuntimeError) as exc_info:
            factory.create_repository(FailingRepository, mock_session)
        
        assert "Repository creation failed for FailingRepository" in str(exc_info.value)

    def test_register_repository_functionality(self):
        """Test repository registration functionality."""
        factory = RepositoryFactory()
        
        # Test registration
        factory.register_repository("test_repo", MockRepository)
        
        assert factory.is_repository_registered("test_repo")
        assert factory.get_registered_repository("test_repo") is MockRepository

    def test_is_repository_registered_false_case(self):
        """Test is_repository_registered returns False for unregistered repos."""
        factory = RepositoryFactory()
        
        result = factory.is_repository_registered("nonexistent_repo")
        assert result is False

    def test_get_registered_repository_none_case(self):
        """Test get_registered_repository returns None for unregistered repos."""
        factory = RepositoryFactory()
        
        result = factory.get_registered_repository("nonexistent_repo")
        assert result is None

    def test_configure_dependencies_functionality(self):
        """Test dependency configuration functionality."""
        factory = RepositoryFactory()
        mock_di = Mock()
        
        # Initially None
        assert factory._dependency_injector is None
        
        # Configure
        factory.configure_dependencies(mock_di)
        
        # Should be set
        assert factory._dependency_injector is mock_di

    def test_list_registered_repositories_functionality(self):
        """Test listing registered repositories."""
        factory = RepositoryFactory()
        
        # Initially empty
        result = factory.list_registered_repositories()
        assert result == []
        
        # Add some registrations
        factory.register_repository("repo1", MockRepository)
        factory.register_repository("repo2", MockRepository)
        
        result = factory.list_registered_repositories()
        assert len(result) == 2
        assert "repo1" in result
        assert "repo2" in result

    def test_clear_registrations_functionality(self):
        """Test clearing repository registrations."""
        factory = RepositoryFactory()
        
        # Add registrations
        factory.register_repository("repo1", MockRepository)
        factory.register_repository("repo2", MockRepository)
        
        # Verify they exist
        assert len(factory.list_registered_repositories()) == 2
        
        # Clear them
        factory.clear_registrations()
        
        # Verify they're gone
        assert len(factory.list_registered_repositories()) == 0
        assert not factory.is_repository_registered("repo1")
        assert not factory.is_repository_registered("repo2")

    def test_error_handling_with_missing_attributes(self):
        """Test error handling when DI resolve fails."""
        # Create DI mock that raises AttributeError on resolve
        mock_di = MagicMock()
        mock_di.resolve = Mock(side_effect=AttributeError("No such attribute"))
        
        factory = RepositoryFactory(dependency_injector=mock_di)
        mock_session = Mock()
        
        # Should fall back to direct instantiation and succeed
        repository = factory.create_repository(MockRepository, mock_session)
        assert isinstance(repository, MockRepository)

    def test_error_handling_with_attribute_error(self):
        """Test error handling when DI raises AttributeError."""
        mock_di = Mock()
        mock_di.resolve = Mock(side_effect=AttributeError("No such attribute"))
        
        factory = RepositoryFactory(dependency_injector=mock_di)
        mock_session = Mock()
        
        # Should fall back to direct instantiation and succeed
        repository = factory.create_repository(MockRepository, mock_session)
        assert isinstance(repository, MockRepository)

    def test_repository_registration_workflow(self):
        """Test complete repository registration and creation workflow."""
        factory = RepositoryFactory()
        
        # Step 1: Register repository
        factory.register_repository("user_repo", MockRepository)
        
        # Step 2: Verify registration
        assert factory.is_repository_registered("user_repo")
        registered_class = factory.get_registered_repository("user_repo")
        assert registered_class is MockRepository
        
        # Step 3: Create instance
        mock_session = Mock()
        repository = factory.create_repository(MockRepository, mock_session)
        
        # Step 4: Verify instance
        assert isinstance(repository, MockRepository)
        assert repository.session is mock_session

    def test_dependency_injection_edge_cases(self):
        """Test DI edge cases to improve coverage."""
        factory = RepositoryFactory()
        
        # Test with DI that resolves to None or fails
        mock_di = Mock()
        mock_di.resolve = Mock(side_effect=KeyError("Not found"))
        
        factory.configure_dependencies(mock_di)
        mock_session = Mock()
        
        # Should fall back to direct instantiation and succeed
        repository = factory.create_repository(MockRepository, mock_session)
        assert isinstance(repository, MockRepository)
        
        # Verify resolve was called
        mock_di.resolve.assert_called_once_with("MockRepository")

    def test_multiple_repository_registrations_same_name(self):
        """Test registering multiple repositories with same name."""
        factory = RepositoryFactory()
        
        class FirstRepo:
            pass
            
        class SecondRepo:
            pass
        
        # Register first
        factory.register_repository("same_name", FirstRepo)
        assert factory.get_registered_repository("same_name") is FirstRepo
        
        # Register second with same name (should overwrite)
        factory.register_repository("same_name", SecondRepo)
        assert factory.get_registered_repository("same_name") is SecondRepo
        
        # Should still only have one entry
        assert len(factory.list_registered_repositories()) == 1

    def test_repository_factory_state_consistency(self):
        """Test that factory maintains consistent state across operations."""
        factory = RepositoryFactory()
        
        # Initial state
        assert len(factory._registered_repositories) == 0
        assert factory._dependency_injector is None
        
        # Add registrations
        for i in range(5):
            factory.register_repository(f"repo_{i}", MockRepository)
        
        assert len(factory._registered_repositories) == 5
        assert len(factory.list_registered_repositories()) == 5
        
        # Configure DI
        mock_di = Mock()
        factory.configure_dependencies(mock_di)
        assert factory._dependency_injector is mock_di
        
        # DI configuration shouldn't affect registrations
        assert len(factory._registered_repositories) == 5
        
        # Clear registrations
        factory.clear_registrations()
        assert len(factory._registered_repositories) == 0
        
        # DI should still be configured
        assert factory._dependency_injector is mock_di