"""Test suite for data dependency injection registration."""

import pytest
from unittest.mock import Mock, patch

from src.core.dependency_injection import DependencyInjector
from src.data.di_registration import configure_data_dependencies, register_data_services


class TestRegisterDataServices:
    """Test suite for register_data_services function."""

    @pytest.fixture
    def mock_injector(self):
        """Create mock dependency injector."""
        injector = Mock(spec=DependencyInjector)
        injector.resolve = Mock()
        injector.register_factory = Mock()
        return injector

    def test_register_data_services_basic(self, mock_injector):
        """Test basic data services registration."""
        # Mock resolve to return None for optional dependencies
        mock_injector.resolve.side_effect = Exception("Not found")
        
        with patch('src.data.storage.database_storage.DatabaseStorage'):
            with patch('src.data.cache.redis_cache.RedisCache'):
                with patch('src.data.validation.market_data_validator.MarketDataValidator'):
                    with patch('src.data.services.refactored_data_service.RefactoredDataService'):
                        register_data_services(mock_injector)
        
        # Verify that register_factory was called for each service
        assert mock_injector.register_factory.call_count >= 4  # At least 4 main services

    def test_database_storage_factory_with_config(self, mock_injector):
        """Test database storage factory with available config."""
        mock_config_service = Mock()
        mock_config = Mock()
        mock_config_service.get_config.return_value = mock_config
        
        mock_database_service = Mock()
        
        def mock_resolve(name):
            if name == "ConfigService":
                return mock_config_service
            elif name == "DatabaseService":
                return mock_database_service
            else:
                raise Exception("Not found")
        
        mock_injector.resolve.side_effect = mock_resolve
        
        with patch('src.data.storage.database_storage.DatabaseStorage') as mock_storage_class:
            mock_storage_instance = Mock()
            mock_storage_class.return_value = mock_storage_instance
            
            register_data_services(mock_injector)
            
            # Get the factory function that was registered
            factory_call = mock_injector.register_factory.call_args_list[0]
            factory_func = factory_call[0][1]  # Second argument is the factory function
            
            # Call the factory function
            result = factory_func()
            
            assert result is mock_storage_instance
            mock_storage_class.assert_called_with(
                config=mock_config,
                database_service=mock_database_service
            )

    def test_database_storage_factory_without_dependencies(self, mock_injector):
        """Test database storage factory without dependencies."""
        mock_injector.resolve.side_effect = Exception("Not found")
        
        with patch('src.data.storage.database_storage.DatabaseStorage') as mock_storage_class:
            with patch('src.core.config.Config') as mock_config_class:
                mock_storage_instance = Mock()
                mock_config_instance = Mock()
                mock_storage_class.return_value = mock_storage_instance
                mock_config_class.return_value = mock_config_instance
                
                register_data_services(mock_injector)
                
                # Get the factory function that was registered
                factory_call = mock_injector.register_factory.call_args_list[0]
                factory_func = factory_call[0][1]
                
                # Call the factory function
                result = factory_func()
                
                assert result is mock_storage_instance
                mock_storage_class.assert_called_with(
                    config=mock_config_instance,
                    database_service=None
                )

    def test_redis_cache_factory(self, mock_injector):
        """Test redis cache factory registration."""
        mock_injector.resolve.side_effect = Exception("Not found")
        
        with patch('src.data.cache.redis_cache.RedisCache') as mock_cache_class:
            with patch('src.core.config.Config') as mock_config_class:
                mock_cache_instance = Mock()
                mock_config_instance = Mock()
                mock_cache_class.return_value = mock_cache_instance
                mock_config_class.return_value = mock_config_instance
                
                register_data_services(mock_injector)
                
                # Get the redis cache factory function
                redis_factory_call = None
                for call in mock_injector.register_factory.call_args_list:
                    if call[0][0] == "DataCacheInterface":
                        redis_factory_call = call
                        break
                
                assert redis_factory_call is not None
                factory_func = redis_factory_call[0][1]
                
                # Call the factory function
                result = factory_func()
                
                assert result is mock_cache_instance
                mock_cache_class.assert_called_with(config=mock_config_instance)

    def test_market_data_validator_factory(self, mock_injector):
        """Test market data validator factory registration."""
        with patch('src.data.validation.market_data_validator.MarketDataValidator') as mock_validator_class:
            mock_validator_instance = Mock()
            mock_validator_class.return_value = mock_validator_instance
            
            register_data_services(mock_injector)
            
            # Get the validator factory function
            validator_factory_call = None
            for call in mock_injector.register_factory.call_args_list:
                if call[0][0] == "DataValidatorInterface":
                    validator_factory_call = call
                    break
            
            assert validator_factory_call is not None
            factory_func = validator_factory_call[0][1]
            
            # Call the factory function
            result = factory_func()
            
            assert result is mock_validator_instance
            mock_validator_class.assert_called_with()

    def test_refactored_data_service_factory(self, mock_injector):
        """Test refactored data service factory registration."""
        mock_storage = Mock()
        mock_cache = Mock()
        mock_validator = Mock()
        
        def mock_resolve(name):
            if name == "DataStorageInterface":
                return mock_storage
            elif name == "DataCacheInterface":
                return mock_cache
            elif name == "ServiceDataValidatorInterface":
                return mock_validator
            else:
                raise Exception("Not found")
        
        mock_injector.resolve.side_effect = mock_resolve
        
        with patch('src.data.services.refactored_data_service.RefactoredDataService') as mock_service_class:
            with patch('src.core.config.Config') as mock_config_class:
                mock_service_instance = Mock()
                mock_config_instance = Mock()
                mock_service_class.return_value = mock_service_instance
                mock_config_class.return_value = mock_config_instance
                
                register_data_services(mock_injector)
                
                # Get the data service factory function
                service_factory_call = None
                for call in mock_injector.register_factory.call_args_list:
                    if call[0][0] == "DataServiceInterface":
                        service_factory_call = call
                        break
                
                assert service_factory_call is not None
                factory_func = service_factory_call[0][1]
                
                # Call the factory function
                result = factory_func()
                
                assert result is mock_service_instance

    def test_registration_error_handling(self, mock_injector):
        """Test error handling in registration process."""
        mock_injector.resolve.side_effect = Exception("Service not found")
        mock_injector.register_factory = Mock(side_effect=Exception("Registration failed"))
        
        # The current implementation propagates exceptions, which is expected behavior
        # Test that the exception is raised when registration fails
        with patch('src.data.storage.database_storage.DatabaseStorage'):
            with patch('src.data.di_registration.logger') as mock_logger:
                with pytest.raises(Exception, match="Registration failed"):
                    register_data_services(mock_injector)


class TestConfigureDataDependencies:
    """Test suite for configure_data_dependencies function."""

    def test_configure_data_dependencies(self):
        """Test configure_data_dependencies creates injector."""
        with patch('src.data.di_registration.DependencyInjector') as mock_injector_class:
            with patch('src.data.di_registration.register_data_services') as mock_register:
                mock_injector_instance = Mock()
                mock_injector_class.return_value = mock_injector_instance
                
                result = configure_data_dependencies()
                
                assert result is mock_injector_instance
                mock_injector_class.assert_called_once()
                mock_register.assert_called_once_with(mock_injector_instance)

    def test_configure_data_dependencies_with_existing_injector(self):
        """Test configure_data_dependencies with existing injector."""
        existing_injector = Mock(spec=DependencyInjector)
        
        with patch('src.data.di_registration.register_data_services') as mock_register:
            result = configure_data_dependencies(existing_injector)
            
            assert result is existing_injector
            mock_register.assert_called_once_with(existing_injector)

    def test_configure_data_dependencies_registration_error(self):
        """Test configure_data_dependencies handles registration errors."""
        with patch('src.data.di_registration.DependencyInjector') as mock_injector_class:
            with patch('src.data.di_registration.register_data_services') as mock_register:
                mock_injector_instance = Mock()
                mock_injector_class.return_value = mock_injector_instance
                mock_register.side_effect = Exception("Registration error")
                
                # configure_data_dependencies doesn't catch registration errors,
                # so it should propagate the exception
                with pytest.raises(Exception, match="Registration error"):
                    configure_data_dependencies()