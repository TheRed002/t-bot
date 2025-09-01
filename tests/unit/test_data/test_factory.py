"""Test suite for data factory."""

import pytest
from unittest.mock import MagicMock, Mock, patch

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
        with patch('src.data.di_registration.configure_data_dependencies', return_value=mock_injector):
            return DataServiceFactory(injector=mock_injector)

    def test_initialization_with_injector(self, mock_injector):
        """Test factory initialization with provided injector."""
        with patch('src.data.di_registration.configure_data_dependencies') as mock_configure:
            factory = DataServiceFactory(injector=mock_injector)
            
            assert factory.injector is mock_injector
            mock_configure.assert_not_called()

    def test_initialization_without_injector(self):
        """Test factory initialization without injector creates one."""
        mock_injector = Mock(spec=DependencyInjector)
        
        with patch('src.data.di_registration.configure_data_dependencies', return_value=mock_injector) as mock_configure:
            factory = DataServiceFactory()
            
            assert factory.injector is mock_injector
            mock_configure.assert_called_once()

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
        
        with patch('src.data.di_registration.configure_data_dependencies', return_value=mock_test_injector):
            result = factory.create_testing_data_service(
                mock_storage=mock_storage,
                mock_cache=mock_cache,
                mock_validator=mock_validator
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
        
        with patch('src.data.di_registration.configure_data_dependencies', return_value=mock_test_injector):
            result = factory.create_testing_data_service()
        
        assert result is mock_data_service
        mock_test_injector.register_factory.assert_not_called()

    def test_create_data_service_from_config(self, mock_data_service):
        """Test creating data service from config."""
        mock_config = Mock()
        mock_metrics = Mock()
        mock_injector = Mock(spec=DependencyInjector)
        mock_injector.resolve.return_value = mock_data_service
        mock_injector.register_factory = Mock()
        
        with patch('src.data.di_registration.configure_data_dependencies', return_value=mock_injector):
            result = DataServiceFactory.create_data_service_from_config(
                config=mock_config,
                use_cache=True,
                use_validator=True,
                metrics_collector=mock_metrics
            )
        
        assert result is mock_data_service
        # Verify that register_factory was called for Config and MetricsCollector
        call_args = [call[0][0] for call in mock_injector.register_factory.call_args_list]
        assert "Config" in call_args
        assert "MetricsCollector" in call_args
        mock_injector.resolve.assert_called_once_with("DataServiceInterface")

    def test_create_data_service_from_config_no_metrics(self, mock_data_service):
        """Test creating data service from config without metrics collector."""
        mock_config = Mock()
        mock_injector = Mock(spec=DependencyInjector)
        mock_injector.resolve.return_value = mock_data_service
        mock_injector.register_factory = Mock()
        
        with patch('src.data.di_registration.configure_data_dependencies', return_value=mock_injector):
            result = DataServiceFactory.create_data_service_from_config(
                config=mock_config,
                use_cache=False,
                use_validator=False
            )
        
        assert result is mock_data_service
        # Verify that register_factory was called for Config
        call_args = [call[0][0] for call in mock_injector.register_factory.call_args_list]
        assert "Config" in call_args
        mock_injector.resolve.assert_called_once_with("DataServiceInterface")