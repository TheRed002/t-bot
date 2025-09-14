"""
Simplified tests for strategies module dependency injection registration.
"""

import pytest
from unittest.mock import MagicMock

from src.core.dependency_injection import DependencyContainer
from src.strategies.di_registration import register_strategies_dependencies
from src.strategies.service import StrategyService
from src.strategies.repository import StrategyRepository
from src.strategies.factory import StrategyFactory


@pytest.fixture
def mock_container():
    """Create a mock dependency container."""
    container = MagicMock(spec=DependencyContainer)
    container.register = MagicMock()
    return container


class TestStrategiesDependencyRegistration:
    """Test dependency injection registration for strategies module."""

    def test_register_strategies_dependencies_success(self, mock_container):
        """Test successful registration of strategies dependencies."""
        # Execute
        register_strategies_dependencies(mock_container)

        # Verify service registrations were called
        assert mock_container.register.call_count == 7

        # Check registration calls
        registration_calls = mock_container.register.call_args_list
        
        # Check that services are registered
        registered_names = [call[1]["name"] for call in registration_calls]
        assert "StrategyService" in registered_names
        assert "StrategyRepository" in registered_names
        assert "StrategyServiceInterface" in registered_names
        assert "StrategyRepositoryInterface" in registered_names

        # Check singleton settings
        service_call = next(call for call in registration_calls if call[1]["name"] == "StrategyService")
        assert service_call[1]["singleton"] is True

        repo_call = next(call for call in registration_calls if call[1]["name"] == "StrategyRepository") 
        assert repo_call[1]["singleton"] is False

    def test_register_strategies_dependencies_container_error(self, mock_container):
        """Test registration handles container errors."""
        # Setup
        mock_container.register.side_effect = Exception("Container error")

        # Execute & Verify
        with pytest.raises(Exception, match="Container error"):
            register_strategies_dependencies(mock_container)

    def test_strategy_service_factory_function(self, mock_container):
        """Test that the strategy service factory function works."""
        # Execute registration to get the factory
        register_strategies_dependencies(mock_container)

        # Get the service factory from the first register call
        service_call = next(
            call for call in mock_container.register.call_args_list 
            if call[1]["name"] == "StrategyService"
        )
        factory = service_call[1]["service"]

        # Test factory creates service instance
        service = factory()
        assert isinstance(service, StrategyService)
        assert service.name == "StrategyService"

        # Test factory with dependencies
        mock_repository = MagicMock()
        mock_risk_manager = MagicMock()
        service_with_deps = factory(
            repository=mock_repository,
            risk_manager=mock_risk_manager,
        )
        
        assert isinstance(service_with_deps, StrategyService)
        assert service_with_deps._repository is mock_repository
        assert service_with_deps._risk_manager is mock_risk_manager

    def test_strategy_repository_factory_function(self, mock_container):
        """Test that the strategy repository factory function works."""
        # Execute registration
        register_strategies_dependencies(mock_container)

        # Get the repository factory
        repo_call = next(
            call for call in mock_container.register.call_args_list 
            if call[1]["name"] == "StrategyRepository"
        )
        factory = repo_call[1]["service"]

        # Test factory creates repository instance
        mock_session = MagicMock()
        repository = factory(session=mock_session)
        
        assert isinstance(repository, StrategyRepository)
        assert repository.session is mock_session

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

        # Verify that registration was called expected number of times
        assert mock_container.register.call_count == 7

    def test_registration_with_different_config(self, mock_container):
        """Test registration works with various configurations."""
        # Execute registration
        register_strategies_dependencies(mock_container)

        # Get service factory and test with different configs
        service_call = next(
            call for call in mock_container.register.call_args_list 
            if call[1]["name"] == "StrategyService"
        )
        factory = service_call[1]["service"]

        # Test with different config types
        configs = [
            None,
            {},
            {"simple": "config"},
            {"complex": {"nested": {"config": [1, 2, 3]}}},
        ]

        for config in configs:
            service = factory(config=config)
            assert isinstance(service, StrategyService)

    def test_factory_parameter_handling(self, mock_container):
        """Test that factory handles all expected parameters."""
        # Execute registration
        register_strategies_dependencies(mock_container)

        # Get service factory
        service_call = next(
            call for call in mock_container.register.call_args_list 
            if call[1]["name"] == "StrategyService"
        )
        factory = service_call[1]["service"]

        # Test with all supported parameters
        test_params = {
            "repository": MagicMock(),
            "risk_manager": MagicMock(),
            "exchange_factory": MagicMock(),
            "data_service": MagicMock(),
            "service_manager": MagicMock(),
            "config": {"test": True},
        }

        # This should not raise any errors
        service = factory(**test_params)
        assert isinstance(service, StrategyService)
        
        # Verify dependencies were injected
        assert service._repository is test_params["repository"]
        assert service._risk_manager is test_params["risk_manager"]
        assert service._exchange_factory is test_params["exchange_factory"]