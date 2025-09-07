"""Web interface services dependency injection registration."""

from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger
from src.web_interface.factory import WebInterfaceFactory

logger = get_logger(__name__)


def register_web_interface_services(injector: DependencyInjector) -> None:
    """
    Register web interface services with the dependency injector using factory pattern.

    Args:
        injector: Dependency injector instance
    """

    # Create factory instance with injector
    factory = WebInterfaceFactory(injector)

    # Register factory itself
    injector.register_service("WebInterfaceFactory", factory, singleton=True)

    # Register all services using factory methods
    injector.register_factory("WebServiceRegistry", factory.create_service_registry, singleton=True)

    injector.register_factory("JWTHandler", factory.create_jwt_handler, singleton=True)

    injector.register_factory("AuthManager", factory.create_auth_manager, singleton=True)

    injector.register_factory("TradingService", factory.create_trading_service, singleton=True)

    injector.register_factory(
        "BotManagementService", factory.create_bot_management_service, singleton=True
    )

    injector.register_factory(
        "MarketDataService", factory.create_market_data_service, singleton=True
    )

    injector.register_factory("PortfolioService", factory.create_portfolio_service, singleton=True)

    injector.register_factory("RiskService", factory.create_risk_service, singleton=True)

    injector.register_factory(
        "StrategyServiceImpl", factory.create_strategy_service, singleton=True
    )

    injector.register_factory("APIFacade", factory.create_api_facade, singleton=True)

    injector.register_factory(
        "UnifiedWebSocketManager", factory.create_websocket_manager, singleton=True
    )

    logger.info("Web interface services registered with dependency injector using factory pattern")
