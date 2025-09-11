"""Test suite for data factory."""

from unittest.mock import Mock, patch

import pytest

from src.core.dependency_injection import DependencyInjector
from src.data.factory import DataServiceFactory
from src.data.interfaces import DataServiceInterface


class TestDataServiceFactory:
    """Test suite for DataServiceFactory."""

    @pytest.fixture
    def mock_injector(self):
        """Create mock dependency injector."""
        injector = Mock(spec=DependencyInjector)
        injector.resolve = Mock()
        return injector

    @pytest.fixture
    def mock_data_service(self):
        """Create mock data service."""
        service = Mock(spec=DataServiceInterface)
        return service

    @pytest.fixture
    def factory(self, mock_injector):
        """Create factory with mock injector."""
        with patch(
            "src.data.di_registration.configure_data_dependencies", return_value=mock_injector
        ):
            return DataServiceFactory(injector=mock_injector)

    def test_initialization_with_injector(self, mock_injector):
        """Test factory initialization with provided injector."""
        factory = DataServiceFactory(injector=mock_injector)

        assert factory._injector is mock_injector

    def test_initialization_without_injector(self):
        """Test factory initialization without injector raises error."""
        from src.core.exceptions import ComponentError

        with pytest.raises(ComponentError, match="Injector must be provided to factory"):
            DataServiceFactory()

    def test_create_data_service(self, factory, mock_injector, mock_data_service):
        """Test creating data service."""
        mock_injector.resolve.return_value = mock_data_service

        result = factory.create_data_service(use_cache=True, use_validator=True)

        assert result is mock_data_service
        mock_injector.resolve.assert_called_once_with("DataServiceInterface")

    def test_create_data_service_with_options(self, factory, mock_injector, mock_data_service):
        """Test creating data service with different options."""
        mock_injector.resolve.return_value = mock_data_service

        result = factory.create_data_service(use_cache=False, use_validator=False)

        assert result is mock_data_service
        mock_injector.resolve.assert_called_once_with("DataServiceInterface")

    def test_create_minimal_data_service(self, factory, mock_injector, mock_data_service):
        """Test creating minimal data service."""
        mock_injector.resolve.return_value = mock_data_service

        result = factory.create_minimal_data_service()

        assert result is mock_data_service
        mock_injector.resolve.assert_called_once_with("DataServiceInterface")

    def test_create_testing_data_service(self, factory, mock_data_service):
        """Test creating testing data service with mocks."""
        mock_storage = Mock()
        mock_cache = Mock()
        mock_validator = Mock()

        mock_test_injector = Mock(spec=DependencyInjector)
        mock_test_injector.resolve.return_value = mock_data_service
        mock_test_injector.register_factory = Mock()

        with patch(
            "src.data.di_registration.configure_data_dependencies", return_value=mock_test_injector
        ):
            result = factory.create_testing_data_service(
                mock_storage=mock_storage, mock_cache=mock_cache, mock_validator=mock_validator
            )

        assert result is mock_data_service
        # Verify that register_factory was called for each mock service
        assert mock_test_injector.register_factory.call_count >= 3
        # Check that the correct service names were registered
        call_args = [call[0][0] for call in mock_test_injector.register_factory.call_args_list]
        assert "DataStorageInterface" in call_args
        assert "DataCacheInterface" in call_args
        assert "DataValidatorInterface" in call_args

    def test_create_testing_data_service_no_mocks(self, factory, mock_data_service):
        """Test creating testing data service without mocks."""
        mock_test_injector = Mock(spec=DependencyInjector)
        mock_test_injector.resolve.return_value = mock_data_service
        mock_test_injector.register_factory = Mock()

        with patch(
            "src.data.di_registration.configure_data_dependencies", return_value=mock_test_injector
        ):
            result = factory.create_testing_data_service()

        assert result is mock_data_service
        mock_test_injector.register_factory.assert_not_called()

    def test_create_default_data_service(self, mock_data_service):
        """Test creating data service using default factory function."""
        from src.data.factory import create_default_data_service

        mock_config = Mock()
        mock_injector = Mock(spec=DependencyInjector)
        mock_injector.resolve.return_value = mock_data_service
        mock_injector.register_factory = Mock()

        with patch(
            "src.data.di_registration.configure_data_dependencies", return_value=mock_injector
        ):
            result = create_default_data_service(
                config=mock_config,
                injector=None,
            )

        assert result is mock_data_service
        # Verify that register_factory was called for Config
        call_args = [call[0][0] for call in mock_injector.register_factory.call_args_list]
        assert "Config" in call_args
        mock_injector.resolve.assert_called_once_with("DataServiceInterface")

    def test_create_additional_services(self, factory, mock_injector):
        """Test creating additional data services."""
        mock_storage = Mock()
        mock_cache = Mock()
        mock_validator = Mock()

        mock_injector.resolve.side_effect = lambda name: {
            "DataStorageInterface": mock_storage,
            "DataCacheInterface": mock_cache,
            "DataValidatorInterface": mock_validator,
        }.get(name)

        storage_result = factory.create_data_storage()
        cache_result = factory.create_data_cache()
        validator_result = factory.create_data_validator()

        assert storage_result is mock_storage
        assert cache_result is mock_cache
        assert validator_result is mock_validator

        assert mock_injector.resolve.call_count == 3
