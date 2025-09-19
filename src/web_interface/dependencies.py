"""
Web Interface Dependencies.

This module provides dependency injection functions for web interface services,
ensuring proper service resolution and lifecycle management.
"""

from typing import TYPE_CHECKING, Any

from src.core.dependency_injection import DependencyInjector, get_global_injector
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.web_interface.services.analytics_service import WebAnalyticsService
    from src.web_interface.services.auth_service import WebAuthService
    from src.web_interface.services.bot_service import WebBotService
    from src.web_interface.services.capital_service import WebCapitalService
    from src.web_interface.services.data_service import WebDataService
    from src.web_interface.services.exchange_service import WebExchangeService
    from src.web_interface.services.monitoring_service import WebMonitoringService
    from src.web_interface.services.portfolio_service import WebPortfolioService
    from src.web_interface.services.risk_service import WebRiskService
    from src.web_interface.services.trading_service import WebTradingService

logger = get_logger(__name__)


def get_web_auth_service() -> "WebAuthService":
    """Get WebAuthService instance from dependency injection container."""
    injector = get_global_injector()

    if not injector.has_service("WebAuthService"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("WebAuthService")


def get_web_analytics_service() -> "WebAnalyticsService":
    """Get WebAnalyticsService instance from dependency injection container."""
    injector = get_global_injector()

    if not injector.has_service("WebAnalyticsService"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("WebAnalyticsService")


def get_web_capital_service() -> "WebCapitalService":
    """Get WebCapitalService instance from dependency injection container."""
    injector = get_global_injector()

    if not injector.has_service("WebCapitalService"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("WebCapitalService")


def get_web_data_service() -> "WebDataService":
    """Get WebDataService instance from dependency injection container."""
    injector = get_global_injector()

    if not injector.has_service("WebDataService"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("WebDataService")


def get_web_exchange_service() -> "WebExchangeService":
    """Get WebExchangeService instance from dependency injection container."""
    injector = get_global_injector()

    if not injector.has_service("WebExchangeService"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("WebExchangeService")


# Existing service functions (maintained for compatibility)
def get_web_portfolio_service() -> "WebPortfolioService":
    """Get WebPortfolioService instance from dependency injection container."""
    injector = get_global_injector()

    if not injector.has_service("WebPortfolioService"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("WebPortfolioService")


def get_web_trading_service() -> "WebTradingService":
    """Get WebTradingService instance from dependency injection container."""
    injector = get_global_injector()

    if not injector.has_service("WebTradingService"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("WebTradingService")


def get_web_bot_service() -> "WebBotService":
    """Get WebBotService instance from dependency injection container."""
    injector = get_global_injector()

    if not injector.has_service("WebBotService"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("WebBotService")


def get_web_monitoring_service() -> "WebMonitoringService":
    """Get WebMonitoringService instance from dependency injection container."""
    injector = get_global_injector()

    if not injector.has_service("WebMonitoringService"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("WebMonitoringService")


def get_web_risk_service() -> "WebRiskService":
    """Get WebRiskService instance from dependency injection container."""
    injector = get_global_injector()

    if not injector.has_service("WebRiskService"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("WebRiskService")


# Service Registry Functions
def get_service_registry() -> Any:
    """Get ServiceRegistry instance from dependency injection container."""
    injector = get_global_injector()

    if not injector.has_service("WebServiceRegistry"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("WebServiceRegistry")


def get_api_facade(injector: DependencyInjector = None) -> Any:
    """Get APIFacade instance from dependency injection container."""
    if injector is None:
        injector = get_global_injector()

    if not injector.has_service("APIFacade"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("APIFacade")


def get_websocket_manager(injector: DependencyInjector = None) -> Any:
    """Get UnifiedWebSocketManager instance from dependency injection container."""
    if injector is None:
        injector = get_global_injector()

    if not injector.has_service("UnifiedWebSocketManager"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("UnifiedWebSocketManager")


# Authentication Functions
def get_jwt_handler(injector: DependencyInjector = None) -> Any:
    """Get JWTHandler instance from dependency injection container."""
    if injector is None:
        injector = get_global_injector()

    if not injector.has_service("JWTHandler"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("JWTHandler")


def get_auth_manager(injector: DependencyInjector = None) -> Any:
    """Get AuthManager instance from dependency injection container."""
    if injector is None:
        injector = get_global_injector()

    if not injector.has_service("AuthManager"):
        from src.web_interface.di_registration import register_web_interface_services

        register_web_interface_services(injector)

    return injector.resolve("AuthManager")


# Service Instance Functions (for FastAPI Depends)
def get_web_auth_service_instance() -> "WebAuthService":
    """Get WebAuthService instance for FastAPI dependency injection."""
    return get_web_auth_service()


def get_web_analytics_service_instance() -> "WebAnalyticsService":
    """Get WebAnalyticsService instance for FastAPI dependency injection."""
    return get_web_analytics_service()


def get_web_capital_service_instance() -> "WebCapitalService":
    """Get WebCapitalService instance for FastAPI dependency injection."""
    return get_web_capital_service()


def get_web_data_service_instance() -> "WebDataService":
    """Get WebDataService instance for FastAPI dependency injection."""
    return get_web_data_service()


def get_web_exchange_service_instance() -> "WebExchangeService":
    """Get WebExchangeService instance for FastAPI dependency injection."""
    return get_web_exchange_service()


def get_web_portfolio_service_instance() -> "WebPortfolioService":
    """Get WebPortfolioService instance for FastAPI dependency injection."""
    return get_web_portfolio_service()


def get_web_trading_service_instance() -> "WebTradingService":
    """Get WebTradingService instance for FastAPI dependency injection."""
    return get_web_trading_service()


def get_web_bot_service_instance() -> "WebBotService":
    """Get WebBotService instance for FastAPI dependency injection."""
    return get_web_bot_service()


def get_web_monitoring_service_instance() -> "WebMonitoringService":
    """Get WebMonitoringService instance for FastAPI dependency injection."""
    return get_web_monitoring_service()


def get_web_risk_service_instance() -> "WebRiskService":
    """Get WebRiskService instance for FastAPI dependency injection."""
    return get_web_risk_service()


# Utility Functions
def ensure_all_services_registered(injector: DependencyInjector = None) -> None:
    """Ensure all web interface services are registered."""
    if injector is None:
        injector = get_global_injector()

    from src.web_interface.di_registration import register_web_interface_services

    register_web_interface_services(injector)

    # Log registered services
    services = [
        "WebAuthService",
        "WebAnalyticsService",
        "WebCapitalService",
        "WebDataService",
        "WebExchangeService",
        "WebPortfolioService",
        "WebTradingService",
        "WebBotService",
        "WebMonitoringService",
        "WebRiskService",
    ]

    registered_count = sum(1 for service in services if injector.has_service(service))

    logger.info(f"Web interface services: {registered_count}/{len(services)} registered")


def get_all_web_services(injector: DependencyInjector = None) -> dict[str, Any]:
    """Get all web services as a dictionary."""
    if injector is None:
        injector = get_global_injector()

    ensure_all_services_registered(injector)

    return {
        "auth": injector.resolve("WebAuthService"),
        "analytics": injector.resolve("WebAnalyticsService"),
        "capital": injector.resolve("WebCapitalService"),
        "data": injector.resolve("WebDataService"),
        "exchange": injector.resolve("WebExchangeService"),
        "portfolio": injector.resolve("WebPortfolioService"),
        "trading": injector.resolve("WebTradingService"),
        "bot": injector.resolve("WebBotService"),
        "monitoring": injector.resolve("WebMonitoringService"),
        "risk": injector.resolve("WebRiskService"),
    }
