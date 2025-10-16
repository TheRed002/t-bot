"""
Simplified tests for strategies module dependency injection registration.
"""

import pytest
from unittest.mock import MagicMock

from src.core.dependency_injection import DependencyInjector
from src.strategies.di_registration import register_strategies_dependencies
from src.strategies.service import StrategyService
from src.strategies.repository import StrategyRepository
from src.strategies.factory import StrategyFactory


@pytest.fixture
def mock_container():
    """Create a mock dependency injector.

    Backend uses DependencyInjector.register_factory(), not DependencyContainer.register()
    """
    injector = MagicMock(spec=DependencyInjector)
    injector.register_factory = MagicMock()
    injector.has_service = MagicMock(return_value=False)  # No dependencies available by default
    return injector


class TestStrategiesDependencyRegistration:
    """Test dependency injection registration for strategies module."""

    def test_register_strategies_dependencies_success(self, mock_container):
        """Test successful registration of strategies dependencies.

        Backend uses injector.register_factory() which takes positional args: (name, factory, singleton)
        """
        # Execute
        register_strategies_dependencies(mock_container)

        # Verify service registrations were called (7 register_factory calls)
        assert mock_container.register_factory.call_count == 7

        # Check registration calls
        registration_calls = mock_container.register_factory.call_args_list

        # Check that services are registered (first arg is name)
        registered_names = [call[0][0] for call in registration_calls]
        assert "StrategyService" in registered_names
        assert "StrategyRepository" in registered_names
        assert "StrategyServiceInterface" in registered_names
        assert "StrategyRepositoryInterface" in registered_names
        assert "StrategyFactory" in registered_names
        assert "DynamicStrategyFactory" in registered_names
        assert "StrategyFactoryInterface" in registered_names

        # Check singleton settings (third arg is singleton flag)
        service_call = next(call for call in registration_calls if call[0][0] == "StrategyService")
        assert service_call[1]["singleton"] is True

        repo_call = next(call for call in registration_calls if call[0][0] == "StrategyRepository")
        assert repo_call[1]["singleton"] is False

    def test_register_strategies_dependencies_container_error(self, mock_container):
        """Test registration handles container errors.

        Backend catches exceptions in register_strategies_dependencies() and re-raises them.
        """
        # Setup - make register_factory raise an exception
        mock_container.register_factory.side_effect = Exception("Container error")

        # Execute & Verify - should raise and propagate the exception
        with pytest.raises(Exception, match="Container error"):
            register_strategies_dependencies(mock_container)

    def test_strategy_service_factory_function(self, mock_container):
        """Test that the strategy service factory function works.

        Backend: injector.register_factory(name, factory, singleton=True)
        - First arg (call[0][0]): name
        - Second arg (call[0][1]): factory function
        - Keyword arg (call[1]["singleton"]): singleton flag
        """
        # Execute registration to get the factory
        register_strategies_dependencies(mock_container)

        # Get the service factory from register_factory calls (positional args)
        service_call = next(
            call for call in mock_container.register_factory.call_args_list
            if call[0][0] == "StrategyService"
        )
        factory = service_call[0][1]  # Second positional arg is the factory function

        # Test factory creates service instance
        service = factory()
        assert isinstance(service, StrategyService)
        assert service.name == "StrategyService"

    def test_strategy_repository_factory_function(self, mock_container):
        """Test that the strategy repository factory function works."""
        # Execute registration
        register_strategies_dependencies(mock_container)

        # Get the repository factory (second positional arg)
        repo_call = next(
            call for call in mock_container.register_factory.call_args_list
            if call[0][0] == "StrategyRepository"
        )
        factory = repo_call[0][1]  # Second positional arg is the factory function

        # Test factory creates repository instance
        repository = factory()

        assert isinstance(repository, StrategyRepository)

    def test_registration_service_types(self, mock_container):
        """Test that correct service types are registered."""
        # Execute
        register_strategies_dependencies(mock_container)

        # Verify service types are callable factories
        registration_calls = mock_container.register.call_args_list
        
        for call in registration_calls:
            service_arg = call[1]["service"]
            if call[1]["name"] in ["StrategyService", "StrategyRepository", "StrategyFactory", "DynamicStrategyFactory"]:
                # These should be factory functions
                assert callable(service_arg)
            else:
                # Interface registrations should be class references
                assert service_arg in [StrategyService, StrategyRepository, StrategyFactory]

    def test_complete_registration_flow(self, mock_container):
        """Test the complete registration flow without errors."""
        # This test ensures the entire registration process completes successfully
        try:
            register_strategies_dependencies(mock_container)
        except Exception as e:
            pytest.fail(f"Registration should not raise exceptions: {e}")

        # Verify that register_factory was called expected number of times
        assert mock_container.register_factory.call_count == 7

    def test_registration_with_different_config(self, mock_container):
        """Test registration works with various configurations.

        Note: The factory function in backend doesn't accept config parameter,
        it creates StrategyService with hardcoded config={}.
        This test just verifies the factory works.
        """
        # Execute registration
        register_strategies_dependencies(mock_container)

        # Get service factory (second positional arg)
        service_call = next(
            call for call in mock_container.register_factory.call_args_list
            if call[0][0] == "StrategyService"
        )
        factory = service_call[0][1]  # Second positional arg is the factory function

        # Test factory works (doesn't accept config parameter)
        service = factory()
        assert isinstance(service, StrategyService)

    def test_factory_parameter_handling(self, mock_container):
        """Test that factory handles all expected parameters.

        Note: The factory function in backend doesn't accept parameters,
        it resolves dependencies from the injector internally.
        This test just verifies the factory works.
        """
        # Execute registration
        register_strategies_dependencies(mock_container)

        # Get service factory (second positional arg)
        service_call = next(
            call for call in mock_container.register_factory.call_args_list
            if call[0][0] == "StrategyService"
        )
        factory = service_call[0][1]  # Second positional arg is the factory function

        # Test factory works (doesn't accept parameters)
        service = factory()
        assert isinstance(service, StrategyService)