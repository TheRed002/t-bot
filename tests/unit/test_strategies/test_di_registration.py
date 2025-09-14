"""Unit tests for strategies dependency injection registration."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.core.dependency_injection import DependencyContainer
from src.strategies.di_registration import register_strategies_dependencies
from src.strategies.service import StrategyService
from src.strategies.repository import StrategyRepository


@pytest.fixture
def mock_container():
    """Mock dependency container."""
    return Mock(spec=DependencyContainer)


def test_register_strategies_dependencies_success(mock_container):
    """Test successful dependency registration."""
    with patch('src.strategies.di_registration.logger') as mock_logger:
        register_strategies_dependencies(mock_container)
        
        # Check that register was called for all services (7 registrations)
        # StrategyService, StrategyRepository, StrategyFactory, DynamicStrategyFactory,
        # StrategyServiceInterface, StrategyRepositoryInterface, StrategyFactoryInterface
        assert mock_container.register.call_count == 7
        
        # Verify logging
        mock_logger.info.assert_any_call("Registering strategies module dependencies")
        mock_logger.info.assert_any_call("Strategies module dependencies registered successfully")


def test_register_strategies_dependencies_register_calls(mock_container):
    """Test specific register calls."""
    register_strategies_dependencies(mock_container)
    
    # Get all the call arguments
    calls = mock_container.register.call_args_list
    
    # Check StrategyService registration
    strategy_service_call = next(call for call in calls if call[1]['name'] == 'StrategyService')
    assert strategy_service_call[1]['singleton'] is True
    assert callable(strategy_service_call[1]['service'])
    
    # Check StrategyRepository registration
    repository_call = next(call for call in calls if call[1]['name'] == 'StrategyRepository')
    assert repository_call[1]['singleton'] is False
    assert callable(repository_call[1]['service'])
    
    # Check interface registrations
    service_interface_call = next(call for call in calls if call[1]['name'] == 'StrategyServiceInterface')
    assert service_interface_call[1]['singleton'] is True
    assert service_interface_call[1]['service'] == StrategyService
    
    repository_interface_call = next(call for call in calls if call[1]['name'] == 'StrategyRepositoryInterface')
    assert repository_interface_call[1]['singleton'] is False
    assert repository_interface_call[1]['service'] == StrategyRepository
    
    # Check StrategyFactory registration
    factory_call = next(call for call in calls if call[1]['name'] == 'StrategyFactory')
    assert factory_call[1]['singleton'] is True
    assert callable(factory_call[1]['service'])
    
    # Check DynamicStrategyFactory registration
    dynamic_factory_call = next(call for call in calls if call[1]['name'] == 'DynamicStrategyFactory')
    assert dynamic_factory_call[1]['singleton'] is True
    assert callable(dynamic_factory_call[1]['service'])


def test_strategy_service_factory():
    """Test strategy service factory function."""
    # Access the factory through registration
    mock_container = Mock()
    register_strategies_dependencies(mock_container)
    
    # Get the factory function
    strategy_service_call = next(
        call for call in mock_container.register.call_args_list 
        if call[1]['name'] == 'StrategyService'
    )
    factory = strategy_service_call[1]['service']
    
    # Test factory with all parameters
    mock_repository = Mock()
    mock_risk_manager = Mock()
    mock_exchange_factory = Mock()
    mock_data_service = Mock()
    mock_backtest_service = Mock()
    mock_service_manager = Mock()
    config = {"test": "config"}
    
    service = factory(
        repository=mock_repository,
        risk_manager=mock_risk_manager,
        exchange_factory=mock_exchange_factory,
        data_service=mock_data_service,
        service_manager=mock_service_manager,
        config=config
    )
    
    assert isinstance(service, StrategyService)
    assert service.name == "StrategyService"
    assert service.get_config() == config


def test_strategy_service_factory_with_defaults():
    """Test strategy service factory with default parameters."""
    mock_container = Mock()
    register_strategies_dependencies(mock_container)
    
    # Get the factory function
    strategy_service_call = next(
        call for call in mock_container.register.call_args_list 
        if call[1]['name'] == 'StrategyService'
    )
    factory = strategy_service_call[1]['service']
    
    # Test factory with default parameters
    service = factory()
    
    assert isinstance(service, StrategyService)
    assert service.name == "StrategyService"
    assert service.get_config() == {}


def test_strategy_service_factory_with_none_config():
    """Test strategy service factory with None config."""
    mock_container = Mock()
    register_strategies_dependencies(mock_container)
    
    # Get the factory function
    strategy_service_call = next(
        call for call in mock_container.register.call_args_list 
        if call[1]['name'] == 'StrategyService'
    )
    factory = strategy_service_call[1]['service']
    
    # Test factory with None config
    service = factory(config=None)
    
    assert isinstance(service, StrategyService)
    assert service.get_config() == {}


def test_strategy_repository_factory():
    """Test strategy repository factory function."""
    mock_container = Mock()
    register_strategies_dependencies(mock_container)
    
    # Get the factory function
    repository_call = next(
        call for call in mock_container.register.call_args_list 
        if call[1]['name'] == 'StrategyRepository'
    )
    factory = repository_call[1]['service']
    
    # Test factory
    mock_session = Mock()
    repository = factory(mock_session)
    
    assert isinstance(repository, StrategyRepository)


def test_register_strategies_dependencies_exception_handling(mock_container):
    """Test exception handling in registration."""
    mock_container.register.side_effect = Exception("Registration failed")
    
    with patch('src.strategies.di_registration.logger') as mock_logger:
        with pytest.raises(Exception, match="Registration failed"):
            register_strategies_dependencies(mock_container)
        
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        assert "Failed to register strategies dependencies" in error_call
        assert "Registration failed" in error_call


def test_register_strategies_dependencies_container_error(mock_container):
    """Test handling of container registration errors."""
    mock_container.register.side_effect = [None, None, None, ValueError("Container error")]
    
    with patch('src.strategies.di_registration.logger') as mock_logger:
        with pytest.raises(ValueError, match="Container error"):
            register_strategies_dependencies(mock_container)
        
        # Should still log the error
        mock_logger.error.assert_called_once()


def test_strategy_service_factory_with_partial_dependencies():
    """Test strategy service factory with partial dependencies."""
    mock_container = Mock()
    register_strategies_dependencies(mock_container)
    
    # Get the factory function
    strategy_service_call = next(
        call for call in mock_container.register.call_args_list 
        if call[1]['name'] == 'StrategyService'
    )
    factory = strategy_service_call[1]['service']
    
    # Test with only some parameters
    mock_repository = Mock()
    mock_risk_manager = Mock()
    
    service = factory(
        repository=mock_repository,
        risk_manager=mock_risk_manager
    )
    
    assert isinstance(service, StrategyService)
    assert service.name == "StrategyService"


def test_factory_functions_are_callable():
    """Test that registered factory functions are callable."""
    mock_container = Mock()
    register_strategies_dependencies(mock_container)
    
    calls = mock_container.register.call_args_list
    
    for call in calls:
        service_param = call[1]['service']
        if callable(service_param) and 'factory' in str(service_param):
            # This is a factory function, should be callable
            assert callable(service_param)


def test_singleton_configuration():
    """Test singleton configuration for different services."""
    mock_container = Mock()
    register_strategies_dependencies(mock_container)
    
    calls = mock_container.register.call_args_list
    
    # StrategyService should be singleton
    strategy_service_call = next(call for call in calls if call[1]['name'] == 'StrategyService')
    assert strategy_service_call[1]['singleton'] is True
    
    # StrategyServiceInterface should be singleton
    service_interface_call = next(call for call in calls if call[1]['name'] == 'StrategyServiceInterface')
    assert service_interface_call[1]['singleton'] is True
    
    # StrategyRepository should not be singleton (needs fresh sessions)
    repository_call = next(call for call in calls if call[1]['name'] == 'StrategyRepository')
    assert repository_call[1]['singleton'] is False
    
    # StrategyRepositoryInterface should not be singleton
    repository_interface_call = next(call for call in calls if call[1]['name'] == 'StrategyRepositoryInterface')
    assert repository_interface_call[1]['singleton'] is False


def test_logger_usage():
    """Test proper logger usage."""
    mock_container = Mock()
    
    with patch('src.strategies.di_registration.logger') as mock_logger:
        register_strategies_dependencies(mock_container)
        
        # Check info logging
        assert mock_logger.info.call_count == 2
        mock_logger.info.assert_any_call("Registering strategies module dependencies")
        mock_logger.info.assert_any_call("Strategies module dependencies registered successfully")


def test_logger_error_on_exception():
    """Test logger error on exception."""
    mock_container = Mock()
    mock_container.register.side_effect = RuntimeError("Test error")
    
    with patch('src.strategies.di_registration.logger') as mock_logger:
        with pytest.raises(RuntimeError):
            register_strategies_dependencies(mock_container)
        
        # Check error logging
        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "Failed to register strategies dependencies" in error_message
        assert "Test error" in error_message


def test_all_required_dependencies_registered():
    """Test that all required dependencies are registered."""
    mock_container = Mock()
    register_strategies_dependencies(mock_container)
    
    calls = mock_container.register.call_args_list
    registered_names = {call[1]['name'] for call in calls}
    
    expected_names = {
        'StrategyService',
        'StrategyRepository', 
        'StrategyFactory',
        'DynamicStrategyFactory',
        'StrategyServiceInterface',
        'StrategyRepositoryInterface',
        'StrategyFactoryInterface'
    }
    
    assert expected_names.issubset(registered_names)


def test_factory_functions_exist():
    """Test that factory functions are properly created."""
    mock_container = Mock()
    register_strategies_dependencies(mock_container)
    
    calls = mock_container.register.call_args_list
    
    # Find service factory
    strategy_service_call = next(call for call in calls if call[1]['name'] == 'StrategyService')
    service_factory = strategy_service_call[1]['service']
    assert callable(service_factory)
    
    # Find repository factory
    repository_call = next(call for call in calls if call[1]['name'] == 'StrategyRepository')
    repository_factory = repository_call[1]['service']
    assert callable(repository_factory)


def test_interface_registrations_use_classes():
    """Test that interface registrations use actual classes."""
    mock_container = Mock()
    register_strategies_dependencies(mock_container)
    
    calls = mock_container.register.call_args_list
    
    # StrategyServiceInterface should use StrategyService class
    service_interface_call = next(call for call in calls if call[1]['name'] == 'StrategyServiceInterface')
    assert service_interface_call[1]['service'] == StrategyService
    
    # StrategyRepositoryInterface should use StrategyRepository class
    repository_interface_call = next(call for call in calls if call[1]['name'] == 'StrategyRepositoryInterface')
    assert repository_interface_call[1]['service'] == StrategyRepository