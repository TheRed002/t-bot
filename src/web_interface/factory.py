"""
Factory for creating web interface components with proper dependency injection.

This module provides factory functions for creating web interface components
following the established service layer patterns and dependency injection.
"""

from typing import Any

from src.core.base import BaseComponent
from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger
from src.web_interface.interfaces import (
    WebBotServiceInterface,
    WebPortfolioServiceInterface,
    WebRiskServiceInterface,
    WebServiceInterface,
    WebStrategyServiceInterface,
    WebTradingServiceInterface,
)

from .auth.auth_manager import AuthManager
from .facade.api_facade import APIFacade
from .facade.service_registry import ServiceRegistry
from .security.jwt_handler import JWTHandler
from .services import (
    WebBotService,
    WebPortfolioService,
    WebRiskService,
    WebStrategyService,
    WebTradingService,
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
        """Create service registry with dependency injection."""
        if self._service_registry is None:
            if self._injector:
                # Try to get existing registry from injector
                try:
                    self._service_registry = self._injector.resolve("ServiceRegistry")
                    logger.debug("Resolved ServiceRegistry from injector")
                except Exception:
                    # Create new registry
                    self._service_registry = ServiceRegistry()
                    logger.info("Created new ServiceRegistry")
            else:
                self._service_registry = ServiceRegistry()
                logger.info("Created ServiceRegistry without injector")
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

    def create_trading_service(self) -> WebTradingServiceInterface:
        """Create trading service with dependencies."""
        trading_facade = None
        if self._injector:
            try:
                trading_facade = self._injector.resolve("TradingFacade")
            except Exception:
                # TradingFacade is optional
                pass

        service = WebTradingService(trading_facade=trading_facade)

        # Configure dependencies if service supports it
        if hasattr(service, "configure_dependencies") and self._injector:
            service.configure_dependencies(self._injector)

        logger.info("Created WebTradingService")
        return service

    def create_bot_management_service(self) -> WebBotServiceInterface:
        """Create bot management service with dependencies."""
        bot_controller = None
        if self._injector:
            try:
                bot_controller = self._injector.resolve("BotManagementController")
            except Exception:
                # BotManagementController is optional
                logger.warning("BotManagementController not available - bot operations will use mock data")

        service = WebBotService(bot_facade=bot_controller)

        # Configure dependencies if service supports it
        if hasattr(service, "configure_dependencies") and self._injector:
            service.configure_dependencies(self._injector)

        logger.info("Created WebBotService")
        return service

    def create_portfolio_service(self) -> WebPortfolioServiceInterface:
        """Create portfolio service with dependencies."""
        portfolio_facade = None
        if self._injector:
            try:
                portfolio_facade = self._injector.resolve("PortfolioFacade")
            except Exception:
                # PortfolioFacade is optional
                pass

        service = WebPortfolioService(portfolio_facade=portfolio_facade)

        # Configure dependencies if service supports it
        if hasattr(service, "configure_dependencies") and self._injector:
            service.configure_dependencies(self._injector)

        logger.info("Created WebPortfolioService")
        return service

    def create_risk_service(self) -> WebRiskServiceInterface:
        """Create risk service with dependencies."""
        risk_facade = None
        if self._injector:
            try:
                risk_facade = self._injector.resolve("RiskFacade")
            except Exception:
                # RiskFacade is optional
                pass

        service = WebRiskService(risk_service=risk_facade)

        # Configure dependencies if service supports it
        if hasattr(service, "configure_dependencies") and self._injector:
            service.configure_dependencies(self._injector)

        logger.info("Created WebRiskService")
        return service

    def create_strategy_service(self) -> WebStrategyServiceInterface:
        """Create strategy service with dependencies."""
        strategy_service = None

        if self._injector:
            try:
                strategy_service = self._injector.resolve("StrategyService")
            except Exception:
                # StrategyService is optional for mock mode
                pass

        service = WebStrategyService(strategy_service=strategy_service)

        # Configure dependencies if service supports it
        if hasattr(service, "configure_dependencies") and self._injector:
            service.configure_dependencies(self._injector)

        logger.info("Created WebStrategyService")
        return service

    def create_market_data_service(self):
        """Create market data service with dependencies."""
        data_service = None
        if self._injector:
            try:
                data_service = self._injector.resolve("DataService")
            except Exception:
                # DataService is optional
                pass

        # Create a mock market data service for now
        class MockMarketDataService:
            def __init__(self, data_service=None):
                self.data_service = data_service

            def configure_dependencies(self, injector):
                """Configure dependencies from injector."""
                try:
                    self.data_service = injector.resolve("DataService")
                except Exception:
                    pass

            async def initialize(self):
                pass

            async def cleanup(self):
                pass

            async def get_market_data(self, symbol: str, exchange: str = "binance"):
                from datetime import datetime, timezone
                from decimal import Decimal

                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "price": Decimal("50000.00"),
                    "bid": Decimal("49995.00"),
                    "ask": Decimal("50005.00"),
                    "volume_24h": Decimal("1000000.00"),
                    "timestamp": datetime.now(timezone.utc),
                }

            async def get_ticker(self, symbol: str):
                """Get ticker data for symbol."""
                from datetime import datetime, timezone
                from decimal import Decimal
                from src.core.types.market import Ticker

                return Ticker(
                    symbol=symbol,
                    bid_price=Decimal("44995.00"),
                    bid_quantity=Decimal("10"),
                    ask_price=Decimal("45005.00"),
                    ask_quantity=Decimal("10"),
                    last_price=Decimal("45000.00"),
                    last_quantity=Decimal("1"),
                    open_price=Decimal("45000.00"),
                    high_price=Decimal("46000.00"),
                    low_price=Decimal("44000.00"),
                    volume=Decimal("1000"),
                    quote_volume=Decimal("45000000.00"),
                    timestamp=datetime.now(timezone.utc),
                    exchange="binance",
                )

        service = MockMarketDataService(data_service=data_service)

        # Configure dependencies if service supports it
        if hasattr(service, "configure_dependencies") and self._injector:
            service.configure_dependencies(self._injector)

        logger.info("Created MockMarketDataService")
        return service

    def create_api_facade(self) -> APIFacade:
        """Create API facade with dependencies using factory pattern."""
        if self._api_facade is None:
            service_registry = None
            if self._injector:
                # Try to resolve service registry from injector first
                try:
                    service_registry = self._injector.resolve("WebServiceRegistry")
                    logger.debug("Resolved WebServiceRegistry from injector")
                except Exception:
                    try:
                        service_registry = self._injector.resolve("ServiceRegistry")
                        logger.debug("Resolved ServiceRegistry from injector")
                    except Exception as e:
                        # Create service registry if not available
                        logger.debug(f"ServiceRegistry not found in injector, creating new: {e}")
                        service_registry = self.create_service_registry()
            else:
                service_registry = self.create_service_registry()

            # Get web services from injector if available
            trading_service = None
            bot_service = None
            portfolio_service = None
            risk_service = None
            strategy_service = None

            if self._injector:
                try:
                    trading_service = self._injector.resolve("WebTradingService")
                except Exception:
                    pass
                try:
                    bot_service = self._injector.resolve("WebBotService")
                except Exception:
                    pass
                try:
                    portfolio_service = self._injector.resolve("WebPortfolioService")
                except Exception:
                    pass
                try:
                    risk_service = self._injector.resolve("WebRiskService")
                except Exception:
                    pass
                try:
                    strategy_service = self._injector.resolve("WebStrategyService")
                except Exception:
                    pass

            # Create API facade with dependency injection
            self._api_facade = APIFacade(
                service_registry=service_registry,
                injector=self._injector,
                trading_service=trading_service,
                bot_service=bot_service,
                portfolio_service=portfolio_service,
                risk_service=risk_service,
                strategy_service=strategy_service,
            )

            # Configure dependencies if facade supports it
            if hasattr(self._api_facade, "configure_dependencies") and self._injector:
                self._api_facade.configure_dependencies(self._injector)

            logger.info("Created APIFacade with factory pattern")
        return self._api_facade

    def create_websocket_manager(self) -> UnifiedWebSocketManager:
        """Create websocket manager with dependencies using factory pattern."""
        api_facade = None
        if self._injector:
            # Try to resolve API facade from injector first
            try:
                api_facade = self._injector.resolve("APIFacade")
                logger.debug("Resolved APIFacade from injector")
            except Exception:
                try:
                    api_facade = self._injector.resolve("WebAPIFacade")
                    logger.debug("Resolved WebAPIFacade from injector")
                except Exception as e:
                    logger.debug(f"APIFacade not found in injector, creating new: {e}")
                    # Create API facade if not available
                    api_facade = self.create_api_facade()
        else:
            api_facade = self.create_api_facade()

        # Create websocket manager with dependency injection
        manager = UnifiedWebSocketManager(api_facade=api_facade)

        # Configure dependencies if manager supports it
        if hasattr(manager, "configure_dependencies") and self._injector:
            manager.configure_dependencies(self._injector)

        logger.info("Created UnifiedWebSocketManager with factory pattern")
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
) -> WebServiceInterface | Any:
    """
    Service locator function to create web interface services using dependency injection.

    This function follows the service locator pattern to decouple service creation
    from direct factory dependencies.

    Args:
        service_type: Type of service to create
        injector: Optional dependency injector
        config: Optional configuration

    Returns:
        Configured service instance implementing appropriate interface

    Raises:
        ValueError: If service type is not recognized
    """
    # Try to get from injector first (preferred)
    if injector:
        service_key = f"Web{service_type}"
        if injector.has_service(service_key):
            return injector.resolve(service_key)

        # Try without Web prefix
        if injector.has_service(service_type):
            return injector.resolve(service_type)

    # Fallback to factory creation
    if not injector:
        # Get global injector if none provided
        from src.core.dependency_injection import get_global_injector

        injector = get_global_injector()

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
        service = creator()

        # Register created service in injector for future use
        if injector:
            try:
                injector.register_service(f"Web{service_type}", service, singleton=True)
            except Exception:
                # Registration might fail if already exists, that's OK
                pass

        return service
    else:
        from src.core.exceptions import ValidationError
        raise ValidationError(f"Unknown service type: {service_type}", field_name="service_type", field_value=service_type)


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
