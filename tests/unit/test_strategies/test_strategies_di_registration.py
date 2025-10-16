"""Unit tests for strategies dependency injection registration.

IMPORTANT: Backend uses DependencyInjector.register_factory() API (not DependencyContainer.register()).
See src/strategies/di_registration.py for actual implementation.

Backend registers 7 services using register_factory():
1. StrategyService (singleton=True)
2. StrategyRepository (singleton=False)
3. StrategyFactory (singleton=True)
4. DynamicStrategyFactory (singleton=True)
5. StrategyServiceInterface (singleton=True)
6. StrategyRepositoryInterface (singleton=False)
7. StrategyFactoryInterface (singleton=True)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.core.dependency_injection import DependencyInjector
from src.strategies.di_registration import register_strategies_dependencies
from src.strategies.service import StrategyService
from src.strategies.repository import StrategyRepository


@pytest.fixture
def mock_injector():
    """Mock dependency injector (DependencyInjector, not DependencyContainer)."""
    injector = Mock(spec=DependencyInjector)
    injector.has_service = Mock(return_value=False)  # Default: no services available
    injector.resolve = Mock(side_effect=Exception("Service not available"))
    return injector


def test_register_strategies_dependencies_success(mock_injector):
    """Test successful dependency registration using register_factory().

    Backend uses injector.register_factory() not container.register().
    See src/strategies/di_registration.py lines 80, 98, 158, 210, 214, 220, 226.
    """
    with patch('src.strategies.di_registration.logger') as mock_logger:
        register_strategies_dependencies(mock_injector)

        # Check that register_factory was called for all services (7 registrations)
        assert mock_injector.register_factory.call_count == 7

        # Verify logging
        mock_logger.info.assert_any_call("Registering strategies module dependencies")
        mock_logger.info.assert_any_call("Strategies module dependencies registered successfully")


def test_register_strategies_dependencies_register_calls(mock_injector):
    """Test specific register_factory calls.

    Backend API: injector.register_factory(name, factory_func, singleton=bool)
    Args are positional: call[0] = (name, factory_func), call[1] = {'singleton': bool}
    """
    register_strategies_dependencies(mock_injector)

    # Get all the call arguments
    calls = mock_injector.register_factory.call_args_list

    # Helper to extract service name from call
    def get_service_name(call):
        # register_factory(name, factory, singleton=bool)
        # Positional: call[0][0] = name
        # Keyword: call[1].get('name') if using kwargs
        return call[0][0] if call[0] else call[1].get('name')

    # Check StrategyService registration
    strategy_service_call = next(call for call in calls if get_service_name(call) == 'StrategyService')
    # singleton is a keyword arg: call[1]['singleton'] or positional call[0][2]
    singleton = strategy_service_call[1].get('singleton', strategy_service_call[0][2] if len(strategy_service_call[0]) > 2 else None)
    assert singleton is True
    # Factory is second positional arg
    factory = strategy_service_call[0][1]
    assert callable(factory)

    # Check StrategyRepository registration
    repository_call = next(call for call in calls if get_service_name(call) == 'StrategyRepository')
    singleton = repository_call[1].get('singleton', repository_call[0][2] if len(repository_call[0]) > 2 else None)
    assert singleton is False
    factory = repository_call[0][1]
    assert callable(factory)

    # Check interface registrations (also use register_factory with lambda factories)
    service_interface_call = next(call for call in calls if get_service_name(call) == 'StrategyServiceInterface')
    singleton = service_interface_call[1].get('singleton', service_interface_call[0][2] if len(service_interface_call[0]) > 2 else None)
    assert singleton is True
    factory = service_interface_call[0][1]
    assert callable(factory)  # Backend uses lambda, not class directly

    repository_interface_call = next(call for call in calls if get_service_name(call) == 'StrategyRepositoryInterface')
    singleton = repository_interface_call[1].get('singleton', repository_interface_call[0][2] if len(repository_interface_call[0]) > 2 else None)
    assert singleton is False
    factory = repository_interface_call[0][1]
    assert callable(factory)

    # Check StrategyFactory registration
    factory_call = next(call for call in calls if get_service_name(call) == 'StrategyFactory')
    singleton = factory_call[1].get('singleton', factory_call[0][2] if len(factory_call[0]) > 2 else None)
    assert singleton is True
    factory = factory_call[0][1]
    assert callable(factory)

    # Check DynamicStrategyFactory registration
    dynamic_factory_call = next(call for call in calls if get_service_name(call) == 'DynamicStrategyFactory')
    singleton = dynamic_factory_call[1].get('singleton', dynamic_factory_call[0][2] if len(dynamic_factory_call[0]) > 2 else None)
    assert singleton is True
    factory = dynamic_factory_call[0][1]
    assert callable(factory)


def test_strategy_service_factory(mock_injector):
    """Test strategy service factory function.

    Backend factory doesn't accept parameters - it resolves dependencies from injector.
    See src/strategies/di_registration.py lines 29-77.
    """
    register_strategies_dependencies(mock_injector)

    # Get the factory function
    calls = mock_injector.register_factory.call_args_list
    strategy_service_call = next(
        call for call in calls
        if call[0][0] == 'StrategyService'
    )
    factory = strategy_service_call[0][1]

    # Backend factory resolves dependencies from injector, doesn't accept params
    # Test that factory creates StrategyService instance
    service = factory()

    assert isinstance(service, StrategyService)
    assert service.name == "StrategyService"


def test_strategy_service_factory_with_defaults(mock_injector):
    """Test strategy service factory with default parameters.

    Backend factory auto-resolves dependencies. If not available, uses None.
    """
    register_strategies_dependencies(mock_injector)

    # Get the factory function
    calls = mock_injector.register_factory.call_args_list
    strategy_service_call = next(
        call for call in calls
        if call[0][0] == 'StrategyService'
    )
    factory = strategy_service_call[0][1]

    # Test factory with default parameters (auto-resolved from injector)
    service = factory()

    assert isinstance(service, StrategyService)
    assert service.name == "StrategyService"


def test_strategy_service_factory_with_none_config(mock_injector):
    """Test strategy service factory handles missing dependencies gracefully.

    Backend uses try/except when resolving optional dependencies.
    """
    # Mock injector has_service returns False, so dependencies will be None
    register_strategies_dependencies(mock_injector)

    # Get the factory function
    calls = mock_injector.register_factory.call_args_list
    strategy_service_call = next(
        call for call in calls
        if call[0][0] == 'StrategyService'
    )
    factory = strategy_service_call[0][1]

    # Factory should handle None dependencies
    service = factory()

    assert isinstance(service, StrategyService)


def test_strategy_repository_factory(mock_injector):
    """Test strategy repository factory function.

    Backend factory resolves DatabaseService and gets session.
    See src/strategies/di_registration.py lines 83-96.
    """
    register_strategies_dependencies(mock_injector)

    # Get the factory function
    calls = mock_injector.register_factory.call_args_list
    repository_call = next(
        call for call in calls
        if call[0][0] == 'StrategyRepository'
    )
    factory = repository_call[0][1]

    # Test factory (backend resolves session from injector)
    repository = factory()

    assert isinstance(repository, StrategyRepository)


def test_register_strategies_dependencies_exception_handling(mock_injector):
    """Test exception handling in registration.

    Backend catches exceptions, logs error, and re-raises.
    See src/strategies/di_registration.py lines 234-236.
    """
    mock_injector.register_factory.side_effect = Exception("Registration failed")

    with patch('src.strategies.di_registration.logger') as mock_logger:
        with pytest.raises(Exception, match="Registration failed"):
            register_strategies_dependencies(mock_injector)

        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        assert "Failed to register strategies dependencies" in error_call


def test_register_strategies_dependencies_container_error(mock_injector):
    """Test handling of injector registration errors.

    Backend will fail on first register_factory error and raise.
    """
    mock_injector.register_factory.side_effect = [None, None, None, ValueError("Injector error")]

    with patch('src.strategies.di_registration.logger') as mock_logger:
        with pytest.raises(ValueError, match="Injector error"):
            register_strategies_dependencies(mock_injector)

        # Should log the error
        mock_logger.error.assert_called_once()


def test_strategy_service_factory_with_partial_dependencies(mock_injector):
    """Test strategy service factory with partial dependencies available.

    Backend checks injector.has_service() and resolves if available.
    """
    # Mock some services as available
    def has_service_side_effect(name):
        return name in ['StrategyRepository', 'RiskService']

    mock_injector.has_service.side_effect = has_service_side_effect
    mock_injector.resolve.side_effect = lambda name: Mock(name=f"Mock{name}")

    register_strategies_dependencies(mock_injector)

    # Get the factory function
    calls = mock_injector.register_factory.call_args_list
    strategy_service_call = next(
        call for call in calls
        if call[0][0] == 'StrategyService'
    )
    factory = strategy_service_call[0][1]

    # Factory should resolve available dependencies
    service = factory()

    assert isinstance(service, StrategyService)
    assert service.name == "StrategyService"


def test_factory_functions_are_callable(mock_injector):
    """Test that registered factory functions are callable."""
    register_strategies_dependencies(mock_injector)

    calls = mock_injector.register_factory.call_args_list

    for call in calls:
        # Factory is second positional arg: call[0][1]
        factory = call[0][1]
        assert callable(factory), f"Factory for {call[0][0]} is not callable"


def test_singleton_configuration(mock_injector):
    """Test singleton configuration for different services."""
    register_strategies_dependencies(mock_injector)

    calls = mock_injector.register_factory.call_args_list

    def get_singleton(call):
        # singleton can be keyword arg or third positional
        return call[1].get('singleton', call[0][2] if len(call[0]) > 2 else None)

    # StrategyService should be singleton
    strategy_service_call = next(call for call in calls if call[0][0] == 'StrategyService')
    assert get_singleton(strategy_service_call) is True

    # StrategyServiceInterface should be singleton
    service_interface_call = next(call for call in calls if call[0][0] == 'StrategyServiceInterface')
    assert get_singleton(service_interface_call) is True

    # StrategyRepository should not be singleton (needs fresh sessions)
    repository_call = next(call for call in calls if call[0][0] == 'StrategyRepository')
    assert get_singleton(repository_call) is False

    # StrategyRepositoryInterface should not be singleton
    repository_interface_call = next(call for call in calls if call[0][0] == 'StrategyRepositoryInterface')
    assert get_singleton(repository_interface_call) is False


def test_logger_usage(mock_injector):
    """Test proper logger usage."""
    with patch('src.strategies.di_registration.logger') as mock_logger:
        register_strategies_dependencies(mock_injector)

        # Check info logging
        assert mock_logger.info.call_count == 2
        mock_logger.info.assert_any_call("Registering strategies module dependencies")
        mock_logger.info.assert_any_call("Strategies module dependencies registered successfully")


def test_logger_error_on_exception(mock_injector):
    """Test logger error on exception."""
    mock_injector.register_factory.side_effect = RuntimeError("Test error")

    with patch('src.strategies.di_registration.logger') as mock_logger:
        with pytest.raises(RuntimeError):
            register_strategies_dependencies(mock_injector)

        # Check error logging
        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "Failed to register strategies dependencies" in error_message


def test_all_required_dependencies_registered(mock_injector):
    """Test that all required dependencies are registered."""
    register_strategies_dependencies(mock_injector)

    calls = mock_injector.register_factory.call_args_list
    # Service name is first positional arg
    registered_names = {call[0][0] for call in calls}

    expected_names = {
        'StrategyService',
        'StrategyRepository',
        'StrategyFactory',
        'DynamicStrategyFactory',
        'StrategyServiceInterface',
        'StrategyRepositoryInterface',
        'StrategyFactoryInterface'
    }

    assert expected_names == registered_names, f"Missing: {expected_names - registered_names}, Extra: {registered_names - expected_names}"


def test_factory_functions_exist(mock_injector):
    """Test that factory functions are properly created."""
    register_strategies_dependencies(mock_injector)

    calls = mock_injector.register_factory.call_args_list

    # Find service factory
    strategy_service_call = next(call for call in calls if call[0][0] == 'StrategyService')
    service_factory = strategy_service_call[0][1]
    assert callable(service_factory)

    # Find repository factory
    repository_call = next(call for call in calls if call[0][0] == 'StrategyRepository')
    repository_factory = repository_call[0][1]
    assert callable(repository_factory)


def test_interface_registrations_use_classes(mock_injector):
    """Test that interface registrations use lambda factories (not direct classes).

    Backend uses lambda factories that create instances:
    - lambda: StrategyService(name="StrategyService", config={})
    - lambda: StrategyRepository(session=None)
    See src/strategies/di_registration.py lines 214-230.
    """
    register_strategies_dependencies(mock_injector)

    calls = mock_injector.register_factory.call_args_list

    # StrategyServiceInterface should use a lambda factory
    service_interface_call = next(call for call in calls if call[0][0] == 'StrategyServiceInterface')
    factory = service_interface_call[0][1]
    assert callable(factory)
    # Factory should create StrategyService instance
    instance = factory()
    assert isinstance(instance, StrategyService)

    # StrategyRepositoryInterface should use a lambda factory
    repository_interface_call = next(call for call in calls if call[0][0] == 'StrategyRepositoryInterface')
    factory = repository_interface_call[0][1]
    assert callable(factory)
    # Factory should create StrategyRepository instance
    instance = factory()
    assert isinstance(instance, StrategyRepository)
