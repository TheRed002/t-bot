"""
Unit tests for repository factory functionality.

Tests the RepositoryFactory class which creates repository instances
using dependency injection patterns.
"""

from unittest.mock import Mock

import pytest

from src.database.repository.base import BaseRepository
from src.database.repository_factory import RepositoryFactory


class MockRepository(BaseRepository):
    """Mock repository for testing."""

    def __init__(self, session):
        self.session = session
        self.test_attr = "initialized"


class TestRepositoryFactory:
    """Test cases for RepositoryFactory class."""

    @pytest.fixture
    def repository_factory(self):
        """Create RepositoryFactory instance for testing."""
        return RepositoryFactory()

    @pytest.fixture
    def repository_factory_with_di(self):
        """Create RepositoryFactory with dependency injection."""
        mock_di = Mock()
        return RepositoryFactory(dependency_injector=mock_di)

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return Mock()

    def test_repository_factory_init_without_di(self):
        """Test RepositoryFactory initialization without dependency injection."""
        factory = RepositoryFactory()

        assert factory._dependency_injector is None
        assert isinstance(factory._registered_repositories, dict)
        assert len(factory._registered_repositories) == 0
        assert factory._logger is not None

    def test_repository_factory_init_with_di(self):
        """Test RepositoryFactory initialization with dependency injection."""
        mock_di = Mock()
        factory = RepositoryFactory(dependency_injector=mock_di)

        assert factory._dependency_injector == mock_di
        assert isinstance(factory._registered_repositories, dict)

    def test_create_repository_basic(self, repository_factory, mock_session):
        """Test basic repository creation without dependency injection."""
        repository = repository_factory.create_repository(MockRepository, mock_session)

        assert isinstance(repository, MockRepository)
        assert repository.session == mock_session
        assert repository.test_attr == "initialized"

    def test_create_repository_with_di_not_registered(
        self, repository_factory_with_di, mock_session
    ):
        """Test repository creation with DI when repository not registered."""
        mock_di = repository_factory_with_di._dependency_injector
        mock_di.resolve = Mock(side_effect=KeyError("Not found"))  # DI resolution fails

        repository = repository_factory_with_di.create_repository(MockRepository, mock_session)

        # Should fallback to direct instantiation
        assert isinstance(repository, MockRepository)
        assert repository.session == mock_session

    def test_create_repository_with_di_registered_direct(
        self, repository_factory_with_di, mock_session
    ):
        """Test repository creation with DI when repository is registered directly."""
        mock_di = repository_factory_with_di._dependency_injector
        mock_repository = MockRepository(mock_session)
        mock_di.resolve = Mock(return_value=mock_repository)

        repository = repository_factory_with_di.create_repository(MockRepository, mock_session)

        assert repository == mock_repository
        mock_di.resolve.assert_called_once_with("MockRepository")

    def test_create_repository_with_di_registered_container(
        self, repository_factory_with_di, mock_session
    ):
        """Test repository creation with DI when repository is resolved successfully."""
        mock_di = repository_factory_with_di._dependency_injector
        mock_repository = MockRepository(mock_session)
        mock_di.resolve = Mock(return_value=mock_repository)

        repository = repository_factory_with_di.create_repository(MockRepository, mock_session)

        assert repository == mock_repository
        mock_di.resolve.assert_called_once_with("MockRepository")

    def test_create_repository_di_resolve_failure(self, repository_factory_with_di, mock_session):
        """Test repository creation when DI resolve fails."""
        mock_di = repository_factory_with_di._dependency_injector
        mock_di.resolve = Mock(side_effect=Exception("DI resolution failed"))

        # Should raise RuntimeError as per the implementation
        with pytest.raises(RuntimeError, match="Repository creation failed"):
            repository_factory_with_di.create_repository(MockRepository, mock_session)

    def test_create_repository_complete_failure(self, repository_factory):
        """Test repository creation when all attempts fail."""

        class BadRepository:
            def __init__(self, session):
                raise Exception("Cannot initialize")

        with pytest.raises(RuntimeError, match="Repository creation failed"):
            repository_factory.create_repository(BadRepository, Mock())

    def test_register_repository(self, repository_factory):
        """Test repository registration."""
        repository_factory.register_repository("test_repo", MockRepository)

        assert "test_repo" in repository_factory._registered_repositories
        assert repository_factory._registered_repositories["test_repo"] == MockRepository

    def test_is_repository_registered(self, repository_factory):
        """Test checking repository registration status."""
        # Initially not registered
        assert not repository_factory.is_repository_registered("test_repo")

        # After registration
        repository_factory.register_repository("test_repo", MockRepository)
        assert repository_factory.is_repository_registered("test_repo")

    def test_get_registered_repository(self, repository_factory):
        """Test getting registered repository class."""
        # Not registered
        assert repository_factory.get_registered_repository("test_repo") is None

        # After registration
        repository_factory.register_repository("test_repo", MockRepository)
        result = repository_factory.get_registered_repository("test_repo")
        assert result == MockRepository

    def test_configure_dependencies(self, repository_factory):
        """Test dependency injection configuration."""
        mock_di = Mock()

        assert repository_factory._dependency_injector is None
        repository_factory.configure_dependencies(mock_di)
        assert repository_factory._dependency_injector == mock_di

    def test_list_registered_repositories(self, repository_factory):
        """Test listing registered repository names."""
        # Initially empty
        assert repository_factory.list_registered_repositories() == []

        # After registrations
        repository_factory.register_repository("repo1", MockRepository)
        repository_factory.register_repository("repo2", MockRepository)

        registered = repository_factory.list_registered_repositories()
        assert len(registered) == 2
        assert "repo1" in registered
        assert "repo2" in registered

    def test_clear_registrations(self, repository_factory):
        """Test clearing all repository registrations."""
        # Add some registrations
        repository_factory.register_repository("repo1", MockRepository)
        repository_factory.register_repository("repo2", MockRepository)

        assert len(repository_factory._registered_repositories) == 2

        # Clear registrations
        repository_factory.clear_registrations()
        assert len(repository_factory._registered_repositories) == 0
        assert repository_factory.list_registered_repositories() == []


class TestRepositoryFactoryErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def repository_factory(self):
        """Create repository factory for error testing."""
        return RepositoryFactory()

    def test_create_repository_with_malformed_di(self, repository_factory):
        """Test repository creation with malformed dependency injector."""
        # Create a malformed DI that raises exception on resolve
        mock_di = Mock()
        mock_di.resolve = Mock(side_effect=AttributeError("No resolve method"))

        repository_factory._dependency_injector = mock_di
        mock_session = Mock()

        # Should fall back to direct instantiation and succeed
        repository = repository_factory.create_repository(MockRepository, mock_session)
        assert isinstance(repository, MockRepository)

    def test_create_repository_di_container_exception(self, repository_factory):
        """Test repository creation when DI resolve throws exception."""
        mock_di = Mock()
        mock_di.resolve = Mock(side_effect=TypeError("Invalid resolve call"))

        repository_factory._dependency_injector = mock_di
        mock_session = Mock()

        # Should fall back to direct instantiation and succeed
        repository = repository_factory.create_repository(MockRepository, mock_session)
        assert isinstance(repository, MockRepository)

    def test_register_repository_edge_cases(self, repository_factory):
        """Test repository registration edge cases."""
        # Test with None class (should work but be useless)
        repository_factory.register_repository("none_repo", None)
        assert repository_factory.is_repository_registered("none_repo")
        assert repository_factory.get_registered_repository("none_repo") is None

        # Test overwriting registration
        repository_factory.register_repository("test_repo", MockRepository)

        class AnotherRepository:
            pass

        repository_factory.register_repository("test_repo", AnotherRepository)
        assert repository_factory.get_registered_repository("test_repo") == AnotherRepository


class TestRepositoryFactoryIntegration:
    """Test integration scenarios with real-like conditions."""

    def test_complete_workflow(self):
        """Test complete repository factory workflow."""
        factory = RepositoryFactory()

        # Register repository
        factory.register_repository("user_repo", MockRepository)

        # Verify registration
        assert factory.is_repository_registered("user_repo")
        assert len(factory.list_registered_repositories()) == 1

        # Create repository instance
        mock_session = Mock()
        repository = factory.create_repository(MockRepository, mock_session)

        assert isinstance(repository, MockRepository)
        assert repository.session == mock_session

        # Clear and verify
        factory.clear_registrations()
        assert len(factory.list_registered_repositories()) == 0

    def test_dependency_injection_lifecycle(self):
        """Test dependency injection configuration lifecycle."""
        factory = RepositoryFactory()

        # Initially no DI
        assert factory._dependency_injector is None

        # Configure DI
        mock_di = Mock()
        factory.configure_dependencies(mock_di)
        assert factory._dependency_injector == mock_di

        # Create repository with DI fallback
        mock_di.resolve = Mock(side_effect=KeyError("Not found"))

        mock_session = Mock()
        repository = factory.create_repository(MockRepository, mock_session)

        assert isinstance(repository, MockRepository)
        mock_di.resolve.assert_called_once_with("MockRepository")

    def test_error_recovery_patterns(self):
        """Test error recovery in various failure scenarios."""
        factory = RepositoryFactory()

        # Create mock DI that fails in different ways
        mock_di = Mock()
        factory.configure_dependencies(mock_di)

        # First attempt: DI resolve throws exception that should be caught
        mock_di.resolve = Mock(side_effect=KeyError("DI failure"))
        mock_session = Mock()

        # Should recover by falling back to direct instantiation
        repository = factory.create_repository(MockRepository, mock_session)
        assert isinstance(repository, MockRepository)

        # Second attempt: Different DI exception
        mock_di.resolve = Mock(side_effect=AttributeError("Resolution failure"))

        # Should recover by falling back to direct instantiation
        repository = factory.create_repository(MockRepository, mock_session)
        assert isinstance(repository, MockRepository)


class TestRepositoryFactoryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_multiple_registrations_same_name(self):
        """Test multiple registrations with same name."""
        factory = RepositoryFactory()

        class FirstRepository:
            pass

        class SecondRepository:
            pass

        factory.register_repository("same_name", FirstRepository)
        factory.register_repository("same_name", SecondRepository)

        # Latest registration should win
        assert factory.get_registered_repository("same_name") == SecondRepository

    def test_empty_string_registration(self):
        """Test registration with empty string name."""
        factory = RepositoryFactory()

        factory.register_repository("", MockRepository)
        assert factory.is_repository_registered("")
        assert factory.get_registered_repository("") == MockRepository

    def test_special_characters_in_names(self):
        """Test registration with special characters in names."""
        factory = RepositoryFactory()

        special_names = ["repo@name", "repo-name", "repo_name", "repo.name", "repo$name"]

        for name in special_names:
            factory.register_repository(name, MockRepository)
            assert factory.is_repository_registered(name)
            assert factory.get_registered_repository(name) == MockRepository

    def test_case_sensitivity(self):
        """Test case sensitivity in repository names."""
        factory = RepositoryFactory()

        factory.register_repository("MyRepo", MockRepository)

        assert factory.is_repository_registered("MyRepo")
        assert not factory.is_repository_registered("myrepo")
        assert not factory.is_repository_registered("MYREPO")

    def test_large_number_of_registrations(self):
        """Test performance with large number of registrations."""
        factory = RepositoryFactory()

        # Register many repositories
        for i in range(1000):
            factory.register_repository(f"repo_{i}", MockRepository)

        assert len(factory.list_registered_repositories()) == 1000
        assert factory.is_repository_registered("repo_500")
        assert factory.get_registered_repository("repo_999") == MockRepository

        # Clear should work efficiently
        factory.clear_registrations()
        assert len(factory.list_registered_repositories()) == 0
