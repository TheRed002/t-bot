"""
Tests for web_interface.factory module.
"""

from unittest.mock import Mock, patch

import pytest

from src.core.dependency_injection import DependencyInjector
from src.web_interface.factory import (
    WebInterfaceFactory,
    create_web_interface_service,
    create_web_interface_stack,
)


@pytest.fixture
def mock_injector():
    """Create a mock dependency injector."""
    injector = Mock(spec=DependencyInjector)
    injector.has_service.return_value = False
    injector.resolve.return_value = Mock()
    return injector


@pytest.fixture
def factory():
    """Create a WebInterfaceFactory instance."""
    return WebInterfaceFactory()


@pytest.fixture
def factory_with_injector(mock_injector):
    """Create a WebInterfaceFactory instance with injector."""
    return WebInterfaceFactory(mock_injector)


class TestWebInterfaceFactory:
    """Tests for WebInterfaceFactory class."""

    def test_init_without_injector(self):
        """Test factory initialization without injector."""
        factory = WebInterfaceFactory()
        assert factory._injector is None
        assert factory._service_registry is None
        assert factory._auth_manager is None
        assert factory._api_facade is None
        assert factory._jwt_handler is None

    def test_init_with_injector(self, mock_injector):
        """Test factory initialization with injector."""
        factory = WebInterfaceFactory(mock_injector)
        assert factory._injector == mock_injector
        assert factory._service_registry is None
        assert factory._auth_manager is None
        assert factory._api_facade is None
        assert factory._jwt_handler is None

    def test_create_service_registry_first_time(self, factory):
        """Test creating service registry for the first time."""
        with patch("src.web_interface.factory.ServiceRegistry") as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry

            result = factory.create_service_registry()

            assert result == mock_registry
            assert factory._service_registry == mock_registry
            mock_registry_class.assert_called_once()

    def test_create_service_registry_cached(self, factory):
        """Test that service registry is cached after first creation."""
        mock_registry = Mock()
        factory._service_registry = mock_registry

        result = factory.create_service_registry()

        assert result == mock_registry

    def test_create_jwt_handler_without_injector(self, factory):
        """Test creating JWT handler without injector."""
        with patch("src.web_interface.factory.JWTHandler") as mock_jwt_class:
            with patch("src.core.config.Config") as mock_config_class:
                mock_jwt = Mock()
                mock_jwt_class.return_value = mock_jwt
                mock_config = Mock()
                mock_config_class.return_value = mock_config

                result = factory.create_jwt_handler()

                assert result == mock_jwt
                assert factory._jwt_handler == mock_jwt
                mock_jwt_class.assert_called_once_with(mock_config)

    def test_create_jwt_handler_with_injector_success(self, factory_with_injector):
        """Test creating JWT handler with injector successfully."""
        mock_config = Mock()
        factory_with_injector._injector.resolve.return_value = mock_config

        with patch("src.web_interface.factory.JWTHandler") as mock_jwt_class:
            mock_jwt = Mock()
            mock_jwt_class.return_value = mock_jwt

            result = factory_with_injector.create_jwt_handler()

            assert result == mock_jwt
            mock_jwt_class.assert_called_once_with(mock_config)

    def test_create_jwt_handler_with_injector_failure(self, factory_with_injector):
        """Test creating JWT handler when injector fails."""
        factory_with_injector._injector.resolve.side_effect = Exception("Injector failed")

        with patch("src.web_interface.factory.JWTHandler") as mock_jwt_class:
            with patch("src.core.config.Config") as mock_config_class:
                mock_jwt = Mock()
                mock_jwt_class.return_value = mock_jwt
                mock_config = Mock()
                mock_config_class.return_value = mock_config

                result = factory_with_injector.create_jwt_handler()

                assert result == mock_jwt
                mock_jwt_class.assert_called_once_with(mock_config)

    def test_create_jwt_handler_cached(self, factory):
        """Test that JWT handler is cached after first creation."""
        mock_jwt = Mock()
        factory._jwt_handler = mock_jwt

        result = factory.create_jwt_handler()

        assert result == mock_jwt

    def test_create_auth_manager_without_injector(self, factory):
        """Test creating auth manager without injector."""
        with patch.object(factory, "create_jwt_handler") as mock_create_jwt:
            with patch("src.web_interface.factory.AuthManager") as mock_auth_class:
                mock_jwt = Mock()
                mock_create_jwt.return_value = mock_jwt
                mock_auth = Mock()
                mock_auth_class.return_value = mock_auth

                config = {"test": "config"}
                result = factory.create_auth_manager(config)

                assert result == mock_auth
                assert factory._auth_manager == mock_auth
                mock_auth_class.assert_called_once_with(jwt_handler=mock_jwt, config=config)

    def test_create_auth_manager_with_injector_success(self, factory_with_injector):
        """Test creating auth manager with injector successfully."""
        mock_jwt = Mock()
        factory_with_injector._injector.resolve.return_value = mock_jwt

        with patch("src.web_interface.factory.AuthManager") as mock_auth_class:
            mock_auth = Mock()
            mock_auth_class.return_value = mock_auth

            result = factory_with_injector.create_auth_manager()

            assert result == mock_auth
            mock_auth_class.assert_called_once_with(jwt_handler=mock_jwt, config={})

    def test_create_auth_manager_with_dependencies_config(self, factory_with_injector):
        """Test creating auth manager with dependency configuration."""
        mock_jwt = Mock()
        factory_with_injector._injector.resolve.return_value = mock_jwt

        with patch("src.web_interface.factory.AuthManager") as mock_auth_class:
            mock_auth = Mock()
            mock_auth.configure_dependencies = Mock()
            mock_auth_class.return_value = mock_auth

            result = factory_with_injector.create_auth_manager()

            assert result == mock_auth
            mock_auth.configure_dependencies.assert_called_once_with(factory_with_injector._injector)

    def test_create_auth_manager_cached(self, factory):
        """Test that auth manager is cached after first creation."""
        mock_auth = Mock()
        factory._auth_manager = mock_auth

        result = factory.create_auth_manager()

        assert result == mock_auth

    def test_create_trading_service(self, factory):
        """Test creating trading service."""
        with patch("src.web_interface.factory.WebTradingService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            result = factory.create_trading_service()

            assert result == mock_service
            mock_service_class.assert_called_once_with(trading_facade=None)

    def test_create_trading_service_with_dependencies(self, factory_with_injector):
        """Test creating trading service with dependency configuration."""
        with patch("src.web_interface.factory.WebTradingService") as mock_service_class:
            mock_service = Mock()
            mock_service.configure_dependencies = Mock()
            mock_service_class.return_value = mock_service

            result = factory_with_injector.create_trading_service()

            assert result == mock_service
            mock_service.configure_dependencies.assert_called_once_with(factory_with_injector._injector)

    def test_create_bot_management_service(self, factory):
        """Test creating bot management service."""
        with patch("src.web_interface.factory.WebBotService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            result = factory.create_bot_management_service()

            assert result == mock_service
            mock_service_class.assert_called_once_with(bot_facade=None)

    def test_create_market_data_service(self, factory):
        """Test creating market data service."""
        result = factory.create_market_data_service()

        # The factory creates a MockMarketDataService inline, so we just check it exists
        assert result is not None
        assert hasattr(result, 'get_market_data')
        assert hasattr(result, 'initialize')
        assert hasattr(result, 'cleanup')

    def test_create_portfolio_service(self, factory):
        """Test creating portfolio service."""
        with patch("src.web_interface.factory.WebPortfolioService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            result = factory.create_portfolio_service()

            assert result == mock_service
            mock_service_class.assert_called_once_with(portfolio_facade=None)

    def test_create_risk_service(self, factory):
        """Test creating risk service."""
        with patch("src.web_interface.factory.WebRiskService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            result = factory.create_risk_service()

            assert result == mock_service
            mock_service_class.assert_called_once_with(risk_facade=None)

    def test_create_strategy_service(self, factory):
        """Test creating strategy service."""
        with patch("src.web_interface.factory.WebStrategyService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            result = factory.create_strategy_service()

            assert result == mock_service
            mock_service_class.assert_called_once_with(strategy_facade=None)

    def test_create_api_facade_without_injector(self, factory):
        """Test creating API facade without injector."""
        with patch.object(factory, "create_service_registry") as mock_create_registry:
            with patch("src.web_interface.factory.APIFacade") as mock_facade_class:
                mock_registry = Mock()
                mock_create_registry.return_value = mock_registry
                mock_facade = Mock()
                mock_facade_class.return_value = mock_facade

                result = factory.create_api_facade()

                assert result == mock_facade
                assert factory._api_facade == mock_facade
                mock_facade_class.assert_called_once_with(
                    service_registry=mock_registry,
                    injector=None,
                    trading_service=None,
                    bot_service=None,
                    portfolio_service=None,
                    risk_service=None,
                    strategy_service=None
                )

    def test_create_api_facade_with_injector_success(self, factory_with_injector):
        """Test creating API facade with injector successfully."""
        mock_registry = Mock()
        factory_with_injector._injector.resolve.return_value = mock_registry

        with patch("src.web_interface.factory.APIFacade") as mock_facade_class:
            mock_facade = Mock()
            mock_facade_class.return_value = mock_facade

            result = factory_with_injector.create_api_facade()

            assert result == mock_facade
            mock_facade_class.assert_called_once_with(
                service_registry=mock_registry,
                injector=factory_with_injector._injector,
                trading_service=mock_registry,  # Since mock returns the same object for all resolve calls
                bot_service=mock_registry,
                portfolio_service=mock_registry,
                risk_service=mock_registry,
                strategy_service=mock_registry
            )

    def test_create_api_facade_cached(self, factory):
        """Test that API facade is cached after first creation."""
        mock_facade = Mock()
        factory._api_facade = mock_facade

        result = factory.create_api_facade()

        assert result == mock_facade

    def test_create_websocket_manager_without_injector(self, factory):
        """Test creating websocket manager without injector."""
        with patch.object(factory, "create_api_facade") as mock_create_facade:
            with patch("src.web_interface.factory.UnifiedWebSocketManager") as mock_manager_class:
                mock_facade = Mock()
                mock_create_facade.return_value = mock_facade
                mock_manager = Mock()
                mock_manager_class.return_value = mock_manager

                result = factory.create_websocket_manager()

                assert result == mock_manager
                mock_manager_class.assert_called_once_with(api_facade=mock_facade)

    def test_create_websocket_manager_with_injector_success(self, factory_with_injector):
        """Test creating websocket manager with injector successfully."""
        mock_facade = Mock()
        factory_with_injector._injector.resolve.return_value = mock_facade

        with patch("src.web_interface.factory.UnifiedWebSocketManager") as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            result = factory_with_injector.create_websocket_manager()

            assert result == mock_manager
            mock_manager_class.assert_called_once_with(api_facade=mock_facade)

    def test_create_complete_web_stack(self, factory):
        """Test creating complete web interface stack."""
        # Mock all creation methods
        mock_components = {
            "service_registry": Mock(),
            "jwt_handler": Mock(),
            "auth_manager": Mock(),
            "trading_service": Mock(),
            "bot_management_service": Mock(),
            "market_data_service": Mock(),
            "portfolio_service": Mock(),
            "risk_service": Mock(),
            "strategy_service": Mock(),
            "api_facade": Mock(),
            "websocket_manager": Mock(),
        }

        with patch.object(factory, "create_service_registry", return_value=mock_components["service_registry"]):
            with patch.object(factory, "create_jwt_handler", return_value=mock_components["jwt_handler"]):
                with patch.object(factory, "create_auth_manager", return_value=mock_components["auth_manager"]):
                    with patch.object(factory, "create_trading_service", return_value=mock_components["trading_service"]):
                        with patch.object(factory, "create_bot_management_service", return_value=mock_components["bot_management_service"]):
                            with patch.object(factory, "create_market_data_service", return_value=mock_components["market_data_service"]):
                                with patch.object(factory, "create_portfolio_service", return_value=mock_components["portfolio_service"]):
                                    with patch.object(factory, "create_risk_service", return_value=mock_components["risk_service"]):
                                        with patch.object(factory, "create_strategy_service", return_value=mock_components["strategy_service"]):
                                            with patch.object(factory, "create_api_facade", return_value=mock_components["api_facade"]):
                                                with patch.object(factory, "create_websocket_manager", return_value=mock_components["websocket_manager"]):

                                                    config = {"test": "config"}
                                                    result = factory.create_complete_web_stack(config)

                                                    assert result == mock_components


class TestCreateWebInterfaceService:
    """Tests for create_web_interface_service function."""

    def test_create_service_from_injector(self, mock_injector):
        """Test creating service from injector when available."""
        mock_service = Mock()
        mock_injector.has_service.return_value = True
        mock_injector.resolve.return_value = mock_service

        result = create_web_interface_service("ServiceRegistry", mock_injector)

        assert result == mock_service
        mock_injector.has_service.assert_called_once_with("WebServiceRegistry")
        mock_injector.resolve.assert_called_once_with("WebServiceRegistry")

    def test_create_service_registry(self):
        """Test creating ServiceRegistry."""
        with patch("src.web_interface.factory.WebInterfaceFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_service_registry.return_value = Mock()
            mock_factory_class.return_value = mock_factory

            result = create_web_interface_service("ServiceRegistry")

            assert result is not None
            mock_factory.create_service_registry.assert_called_once()

    def test_create_jwt_handler(self):
        """Test creating JWTHandler."""
        with patch("src.web_interface.factory.WebInterfaceFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_jwt_handler.return_value = Mock()
            mock_factory_class.return_value = mock_factory

            config = {"test": "config"}
            result = create_web_interface_service("JWTHandler", config=config)

            assert result is not None
            mock_factory.create_jwt_handler.assert_called_once_with(config)

    def test_create_auth_manager(self):
        """Test creating AuthManager."""
        with patch("src.web_interface.factory.WebInterfaceFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_auth_manager.return_value = Mock()
            mock_factory_class.return_value = mock_factory

            config = {"test": "config"}
            result = create_web_interface_service("AuthManager", config=config)

            assert result is not None
            mock_factory.create_auth_manager.assert_called_once_with(config)

    def test_create_trading_service(self):
        """Test creating TradingService."""
        with patch("src.web_interface.factory.WebInterfaceFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_trading_service.return_value = Mock()
            mock_factory_class.return_value = mock_factory

            result = create_web_interface_service("TradingService")

            assert result is not None
            mock_factory.create_trading_service.assert_called_once()

    def test_create_bot_management_service(self):
        """Test creating BotManagementService."""
        with patch("src.web_interface.factory.WebInterfaceFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_bot_management_service.return_value = Mock()
            mock_factory_class.return_value = mock_factory

            result = create_web_interface_service("BotManagementService")

            assert result is not None
            mock_factory.create_bot_management_service.assert_called_once()

    def test_create_market_data_service(self):
        """Test creating MarketDataService."""
        with patch("src.web_interface.factory.WebInterfaceFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_market_data_service.return_value = Mock()
            mock_factory_class.return_value = mock_factory

            result = create_web_interface_service("MarketDataService")

            assert result is not None
            mock_factory.create_market_data_service.assert_called_once()

    def test_create_portfolio_service(self):
        """Test creating PortfolioService."""
        with patch("src.web_interface.factory.WebInterfaceFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_portfolio_service.return_value = Mock()
            mock_factory_class.return_value = mock_factory

            result = create_web_interface_service("PortfolioService")

            assert result is not None
            mock_factory.create_portfolio_service.assert_called_once()

    def test_create_risk_service(self):
        """Test creating RiskService."""
        with patch("src.web_interface.factory.WebInterfaceFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_risk_service.return_value = Mock()
            mock_factory_class.return_value = mock_factory

            result = create_web_interface_service("RiskService")

            assert result is not None
            mock_factory.create_risk_service.assert_called_once()

    def test_create_strategy_service(self):
        """Test creating StrategyService."""
        with patch("src.web_interface.factory.WebInterfaceFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_strategy_service.return_value = Mock()
            mock_factory_class.return_value = mock_factory

            result = create_web_interface_service("StrategyService")

            assert result is not None
            mock_factory.create_strategy_service.assert_called_once()

    def test_create_api_facade(self):
        """Test creating APIFacade."""
        with patch("src.web_interface.factory.WebInterfaceFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_api_facade.return_value = Mock()
            mock_factory_class.return_value = mock_factory

            result = create_web_interface_service("APIFacade")

            assert result is not None
            mock_factory.create_api_facade.assert_called_once()

    def test_create_websocket_manager(self):
        """Test creating WebSocketManager."""
        with patch("src.web_interface.factory.WebInterfaceFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_websocket_manager.return_value = Mock()
            mock_factory_class.return_value = mock_factory

            result = create_web_interface_service("WebSocketManager")

            assert result is not None
            mock_factory.create_websocket_manager.assert_called_once()

    def test_create_unknown_service(self):
        """Test creating unknown service type raises error."""
        with pytest.raises(ValueError, match="Unknown service type: UnknownService"):
            create_web_interface_service("UnknownService")


class TestCreateWebInterfaceStack:
    """Tests for create_web_interface_stack function."""

    def test_create_web_interface_stack(self, mock_injector):
        """Test creating complete web interface stack."""
        mock_components = {"test": "components"}

        with patch("src.web_interface.factory.WebInterfaceFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_complete_web_stack.return_value = mock_components
            mock_factory_class.return_value = mock_factory

            config = {"test": "config"}
            result = create_web_interface_stack(mock_injector, config)

            assert result == mock_components
            mock_factory_class.assert_called_once_with(mock_injector)
            mock_factory.create_complete_web_stack.assert_called_once_with(config)

    def test_create_web_interface_stack_without_injector(self):
        """Test creating web interface stack without injector."""
        mock_components = {"test": "components"}

        with patch("src.web_interface.factory.WebInterfaceFactory") as mock_factory_class:
            mock_factory = Mock()
            mock_factory.create_complete_web_stack.return_value = mock_components
            mock_factory_class.return_value = mock_factory

            result = create_web_interface_stack()

            assert result == mock_components
            mock_factory_class.assert_called_once_with(None)
            mock_factory.create_complete_web_stack.assert_called_once_with(None)