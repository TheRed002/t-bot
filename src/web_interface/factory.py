"""
Factory for creating web interface components with proper dependency injection.

This module provides factory functions for creating web interface components
following the established service layer patterns and dependency injection.
"""

from typing import Any

from src.core.base import BaseComponent
from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger

from .auth.auth_manager import AuthManager
from .facade.api_facade import APIFacade
from .facade.service_registry import ServiceRegistry
from .security.jwt_handler import JWTHandler
from .services.service_implementations import (
    BotManagementServiceImpl,
    MarketDataServiceImpl,
    PortfolioServiceImpl,
    RiskServiceImpl,
    StrategyServiceImpl,
    TradingServiceImpl,
)
from .websockets.unified_manager import UnifiedWebSocketManager

logger = get_logger(__name__)


class WebInterfaceFactory(BaseComponent):
    """
    Factory for creating web interface components.

    Provides centralized creation of web services, facades, and managers
    with proper dependency injection.
    """

    def __init__(self, injector: DependencyInjector | None = None) -> None:
        """
        Initialize web interface factory.

        Args:
            injector: Dependency injector for service resolution
        """
        super().__init__()
        self._injector = injector
        self._service_registry: ServiceRegistry | None = None
        self._auth_manager: AuthManager | None = None
        self._api_facade: APIFacade | None = None
        self._jwt_handler: JWTHandler | None = None

    def create_service_registry(self) -> ServiceRegistry:
        """Create service registry."""
        if self._service_registry is None:
            self._service_registry = ServiceRegistry()
            logger.info("Created ServiceRegistry")
        return self._service_registry

    def create_jwt_handler(self, config: dict[str, Any] | None = None) -> JWTHandler:
        """Create JWT handler with configuration."""
        if self._jwt_handler is None:
            if self._injector:
                try:
                    # Try to get config from injector
                    app_config = self._injector.resolve("Config")
                    self._jwt_handler = JWTHandler(app_config)
                except Exception as e:
                    # Fallback with minimal config
                    self.logger.warning(f"Failed to get config from injector: {e}")
                    from src.core.config import Config

                    fallback_config = Config()
                    self._jwt_handler = JWTHandler(fallback_config)
            else:
                # Direct instantiation
                from src.core.config import Config

                fallback_config = Config()
                self._jwt_handler = JWTHandler(fallback_config)
            logger.info("Created JWTHandler")
        return self._jwt_handler

    def create_auth_manager(self, config: dict[str, Any] | None = None) -> AuthManager:
        """Create authentication manager with dependencies."""
        if self._auth_manager is None:
            jwt_handler = None
            if self._injector:
                try:
                    jwt_handler = self._injector.resolve("JWTHandler")
                except Exception as e:
                    # Create JWT handler if not available
                    self.logger.debug(f"JWTHandler not found in injector: {e}")
                    jwt_handler = self.create_jwt_handler(config)
            else:
                jwt_handler = self.create_jwt_handler(config)

            self._auth_manager = AuthManager(jwt_handler=jwt_handler, config=config or {})

            # Configure dependencies if injector is available
            if self._injector and hasattr(self._auth_manager, "configure_dependencies"):
                self._auth_manager.configure_dependencies(self._injector)

            logger.info("Created AuthManager")
        return self._auth_manager

    def create_trading_service(self) -> TradingServiceImpl:
        """Create trading service with dependencies."""
        service = TradingServiceImpl()
        if self._injector and hasattr(service, "configure_dependencies"):
            service.configure_dependencies(self._injector)
        logger.info("Created TradingService")
        return service

    def create_bot_management_service(self) -> BotManagementServiceImpl:
        """Create bot management service with dependencies."""
        service = BotManagementServiceImpl()
        if self._injector and hasattr(service, "configure_dependencies"):
            service.configure_dependencies(self._injector)
        logger.info("Created BotManagementService")
        return service

    def create_market_data_service(self) -> MarketDataServiceImpl:
        """Create market data service with dependencies."""
        service = MarketDataServiceImpl()
        if self._injector and hasattr(service, "configure_dependencies"):
            service.configure_dependencies(self._injector)
        logger.info("Created MarketDataService")
        return service

    def create_portfolio_service(self) -> PortfolioServiceImpl:
        """Create portfolio service with dependencies."""
        service = PortfolioServiceImpl()
        if self._injector and hasattr(service, "configure_dependencies"):
            service.configure_dependencies(self._injector)
        logger.info("Created PortfolioService")
        return service

    def create_risk_service(self) -> RiskServiceImpl:
        """Create risk service with dependencies."""
        service = RiskServiceImpl()
        if self._injector and hasattr(service, "configure_dependencies"):
            service.configure_dependencies(self._injector)
        logger.info("Created RiskService")
        return service

    def create_strategy_service(self) -> StrategyServiceImpl:
        """Create strategy service with dependencies."""
        service = StrategyServiceImpl()
        if self._injector and hasattr(service, "configure_dependencies"):
            service.configure_dependencies(self._injector)
        logger.info("Created StrategyService")
        return service

    def create_api_facade(self) -> APIFacade:
        """Create API facade with dependencies."""
        if self._api_facade is None:
            service_registry = None
            if self._injector:
                try:
                    service_registry = self._injector.resolve("WebServiceRegistry")
                except Exception as e:
                    # Create service registry if not available
                    self.logger.debug(f"WebServiceRegistry not found in injector: {e}")
                    service_registry = self.create_service_registry()
            else:
                service_registry = self.create_service_registry()

            # Create API facade with optional injector
            if self._injector:
                self._api_facade = APIFacade(
                    service_registry=service_registry, injector=self._injector
                )
            else:
                self._api_facade = APIFacade(service_registry=service_registry)
            logger.info("Created APIFacade")
        return self._api_facade

    def create_websocket_manager(self) -> UnifiedWebSocketManager:
        """Create websocket manager with dependencies."""
        api_facade = None
        if self._injector:
            try:
                api_facade = self._injector.resolve("APIFacade")
            except Exception as e:
                self.logger.debug(f"APIFacade not found in injector: {e}")
                # Create API facade if not available
                api_facade = self.create_api_facade()
        else:
            api_facade = self.create_api_facade()

        manager = UnifiedWebSocketManager(api_facade=api_facade)
        logger.info("Created UnifiedWebSocketManager")
        return manager

    def create_complete_web_stack(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Create complete web interface stack with all components.

        Args:
            config: Optional configuration for components

        Returns:
            Dictionary containing all web interface components
        """
        components = {}

        # Core components
        components["service_registry"] = self.create_service_registry()
        components["jwt_handler"] = self.create_jwt_handler(config)
        components["auth_manager"] = self.create_auth_manager(config)

        # Service implementations
        components["trading_service"] = self.create_trading_service()
        components["bot_management_service"] = self.create_bot_management_service()
        components["market_data_service"] = self.create_market_data_service()
        components["portfolio_service"] = self.create_portfolio_service()
        components["risk_service"] = self.create_risk_service()
        components["strategy_service"] = self.create_strategy_service()

        # High-level components
        components["api_facade"] = self.create_api_facade()
        components["websocket_manager"] = self.create_websocket_manager()

        logger.info("Created complete web interface stack")
        return components


def create_web_interface_service(
    service_type: str,
    injector: DependencyInjector | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    """
    Convenience function to create web interface services.

    Args:
        service_type: Type of service to create
        injector: Optional dependency injector
        config: Optional configuration

    Returns:
        Configured service instance
    """
    if injector and injector.has_service(f"Web{service_type}"):
        return injector.resolve(f"Web{service_type}")

    factory = WebInterfaceFactory(injector)

    service_creators = {
        "ServiceRegistry": factory.create_service_registry,
        "JWTHandler": lambda: factory.create_jwt_handler(config),
        "AuthManager": lambda: factory.create_auth_manager(config),
        "TradingService": factory.create_trading_service,
        "BotManagementService": factory.create_bot_management_service,
        "MarketDataService": factory.create_market_data_service,
        "PortfolioService": factory.create_portfolio_service,
        "RiskService": factory.create_risk_service,
        "StrategyService": factory.create_strategy_service,
        "APIFacade": factory.create_api_facade,
        "WebSocketManager": factory.create_websocket_manager,
    }

    creator = service_creators.get(service_type)
    if creator:
        return creator()
    else:
        raise ValueError(f"Unknown service type: {service_type}")


def create_web_interface_stack(
    injector: DependencyInjector | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convenience function to create complete web interface stack.

    Args:
        injector: Optional dependency injector
        config: Optional configuration

    Returns:
        Dictionary containing all web interface components
    """
    factory = WebInterfaceFactory(injector)
    return factory.create_complete_web_stack(config)
