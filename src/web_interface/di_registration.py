"""Web interface services dependency injection registration."""

from typing import TYPE_CHECKING

from src.core.dependency_injection import DependencyInjector
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.web_interface.factory import WebInterfaceFactory
    from src.web_interface.services.bot_service import WebBotService
    from src.web_interface.services.monitoring_service import WebMonitoringService
    from src.web_interface.services.portfolio_service import WebPortfolioService
    from src.web_interface.services.risk_service import WebRiskService
    from src.web_interface.services.trading_service import WebTradingService

logger = get_logger(__name__)


def _register_core_services(injector: DependencyInjector, factory: "WebInterfaceFactory") -> None:
    """Register core service implementations."""
    # Core service implementations - register both with and without Web prefix
    injector.register_factory("TradingService", factory.create_trading_service, singleton=True)
    injector.register_factory(
        "WebTradingServiceImpl", factory.create_trading_service, singleton=True
    )

    injector.register_factory(
        "BotManagementService", factory.create_bot_management_service, singleton=True
    )
    injector.register_factory(
        "WebBotManagementService", factory.create_bot_management_service, singleton=True
    )

    injector.register_factory(
        "MarketDataService", factory.create_market_data_service, singleton=True
    )
    injector.register_factory(
        "WebMarketDataService", factory.create_market_data_service, singleton=True
    )

    injector.register_factory("PortfolioService", factory.create_portfolio_service, singleton=True)
    injector.register_factory(
        "WebPortfolioServiceImpl", factory.create_portfolio_service, singleton=True
    )

    injector.register_factory("RiskService", factory.create_risk_service, singleton=True)
    injector.register_factory("WebRiskServiceImpl", factory.create_risk_service, singleton=True)

    injector.register_factory("StrategyService", factory.create_strategy_service, singleton=True)
    injector.register_factory(
        "WebStrategyServiceImpl", factory.create_strategy_service, singleton=True
    )

    injector.register_factory("APIFacade", factory.create_api_facade, singleton=True)
    injector.register_factory("WebAPIFacade", factory.create_api_facade, singleton=True)

    injector.register_factory(
        "UnifiedWebSocketManager", factory.create_websocket_manager, singleton=True
    )
    injector.register_factory("WebSocketManager", factory.create_websocket_manager, singleton=True)


def _register_web_business_services(
    injector: DependencyInjector, factory: "WebInterfaceFactory"
) -> None:
    """Register web-specific business logic services."""

    def _create_web_portfolio_service():
        """Factory function for WebPortfolioService with DI."""
        from src.web_interface.services.portfolio_service import WebPortfolioService

        api_facade = (
            injector.resolve("APIFacade")
            if injector.has_service("APIFacade")
            else factory.create_api_facade()
        )
        return WebPortfolioService(portfolio_facade=api_facade)

    def _create_web_trading_service():
        """Factory function for WebTradingService with DI."""
        from src.web_interface.services.trading_service import WebTradingService

        api_facade = (
            injector.resolve("APIFacade")
            if injector.has_service("APIFacade")
            else factory.create_api_facade()
        )
        return WebTradingService(trading_facade=api_facade)

    def _create_web_bot_service():
        """Factory function for WebBotService with DI."""
        from src.web_interface.services.bot_service import WebBotService

        api_facade = (
            injector.resolve("APIFacade")
            if injector.has_service("APIFacade")
            else factory.create_api_facade()
        )
        return WebBotService(bot_facade=api_facade)

    def _create_web_monitoring_service():
        """Factory function for WebMonitoringService with DI."""
        from src.web_interface.services.monitoring_service import WebMonitoringService

        api_facade = (
            injector.resolve("APIFacade")
            if injector.has_service("APIFacade")
            else factory.create_api_facade()
        )
        return WebMonitoringService(monitoring_facade=api_facade)

    def _create_web_risk_service():
        """Factory function for WebRiskService with DI."""
        from src.web_interface.services.risk_service import WebRiskService

        api_facade = (
            injector.resolve("APIFacade")
            if injector.has_service("APIFacade")
            else factory.create_api_facade()
        )
        return WebRiskService(risk_facade=api_facade)

    # Register using factory functions
    injector.register_factory("WebPortfolioService", _create_web_portfolio_service, singleton=True)
    injector.register_factory("WebTradingService", _create_web_trading_service, singleton=True)
    injector.register_factory("WebBotService", _create_web_bot_service, singleton=True)
    injector.register_factory(
        "WebMonitoringService", _create_web_monitoring_service, singleton=True
    )
    injector.register_factory("WebRiskService", _create_web_risk_service, singleton=True)


def register_web_interface_services(injector: DependencyInjector) -> None:
    """
    Register web interface services using factory pattern with dependency injection.

    This registration follows the factory pattern where:
    1. Factory is registered as a singleton
    2. Services are created using factory methods
    3. Dependencies are injected automatically
    4. Services implement defined interfaces

    Args:
        injector: Dependency injector instance
    """
    # Create factory instance with injector
    from src.web_interface.factory import WebInterfaceFactory

    factory = WebInterfaceFactory(injector)

    # Register factory itself
    injector.register_service("WebInterfaceFactory", factory, singleton=True)

    # Register basic infrastructure services
    injector.register_factory("WebServiceRegistry", factory.create_service_registry, singleton=True)
    injector.register_factory("ServiceRegistry", factory.create_service_registry, singleton=True)
    injector.register_factory("JWTHandler", factory.create_jwt_handler, singleton=True)
    injector.register_factory("WebJWTHandler", factory.create_jwt_handler, singleton=True)
    injector.register_factory("AuthManager", factory.create_auth_manager, singleton=True)
    injector.register_factory("WebAuthManager", factory.create_auth_manager, singleton=True)

    # Register core services
    _register_core_services(injector, factory)

    # Register web business services
    _register_web_business_services(injector, factory)

    # Register additional services
    _register_analytics_services(injector)
    _register_utility_services(injector, factory)

    logger.info("Web interface services registered with dependency injector using factory pattern")


def _create_mock_analytics_service():
    """Create mock analytics service for web interface."""

    class MockAnalyticsService:
        """Mock analytics service for testing and development."""

        async def get_portfolio_metrics(self):
            """Get mock portfolio metrics."""
            from datetime import datetime
            from decimal import Decimal

            class MockMetrics:
                """Mock metrics data structure."""

                def __init__(self):
                    self.total_value = Decimal("10000.00")
                    self.total_pnl = Decimal("500.00")
                    self.total_pnl_percentage = Decimal("5.0")
                    self.win_rate = Decimal("0.75")
                    self.sharpe_ratio = Decimal("1.2")
                    self.max_drawdown = Decimal("0.15")
                    self.positions_count = 5
                    self.active_strategies = 3
                    self.timestamp = datetime.utcnow()

            return MockMetrics()

        async def get_risk_metrics(self):
            """Get mock risk metrics."""
            from datetime import datetime
            from decimal import Decimal

            class MockRiskMetrics:
                """Mock risk metrics data structure."""

                def __init__(self):
                    self.portfolio_var = {"95": Decimal("1000.00"), "99": Decimal("1500.00")}
                    self.portfolio_volatility = Decimal("0.25")
                    self.portfolio_beta = Decimal("1.1")
                    self.correlation_risk = Decimal("0.3")
                    self.concentration_risk = Decimal("0.4")
                    self.leverage_ratio = Decimal("2.0")
                    self.margin_usage = Decimal("0.6")
                    self.timestamp = datetime.utcnow()

            return MockRiskMetrics()

        async def get_strategy_metrics(self, strategy=None):
            """Get mock strategy metrics."""
            return None  # No strategies available

    return MockAnalyticsService()


def _resolve_analytics_dependencies(injector: DependencyInjector) -> dict:
    """Resolve analytics service dependencies."""
    return {
        "portfolio_service": (
            injector.resolve("PortfolioServiceProtocol")
            if injector.has_service("PortfolioServiceProtocol")
            else None
        ),
        "risk_service": (
            injector.resolve("RiskServiceProtocol")
            if injector.has_service("RiskServiceProtocol")
            else None
        ),
        "reporting_service": (
            injector.resolve("ReportingServiceProtocol")
            if injector.has_service("ReportingServiceProtocol")
            else None
        ),
        "alert_service": (
            injector.resolve("AlertServiceProtocol")
            if injector.has_service("AlertServiceProtocol")
            else None
        ),
        "operational_service": (
            injector.resolve("OperationalServiceProtocol")
            if injector.has_service("OperationalServiceProtocol")
            else None
        ),
        "export_service": (
            injector.resolve("ExportServiceProtocol")
            if injector.has_service("ExportServiceProtocol")
            else None
        ),
    }


def _register_analytics_services(injector: DependencyInjector) -> None:
    """Register analytics and data services."""

    def _create_web_analytics_service():
        """Factory function for WebAnalyticsService with DI."""
        from src.web_interface.services.analytics_service import WebAnalyticsService

        # Get analytics service or fallback to mock
        analytics_service = (
            injector.resolve("AnalyticsService")
            if injector.has_service("AnalyticsService")
            else _create_mock_analytics_service()
        )

        # Get supporting services
        services = _resolve_analytics_dependencies(injector)

        # Create service instance
        return WebAnalyticsService(analytics_service=analytics_service, **services)

    injector.register_factory("WebAnalyticsService", _create_web_analytics_service, singleton=True)

    def _create_web_capital_service():
        """Factory function for WebCapitalService with DI."""
        capital_service = (
            injector.resolve("CapitalService") if injector.has_service("CapitalService") else None
        )
        currency_service = (
            injector.resolve("AbstractCurrencyManagementService")
            if injector.has_service("AbstractCurrencyManagementService")
            else None
        )
        fund_flow_service = (
            injector.resolve("AbstractFundFlowManagementService")
            if injector.has_service("AbstractFundFlowManagementService")
            else None
        )

        if capital_service is None:
            # Create a mock capital service for the web interface
            class MockCapitalService:
                """Mock capital service for testing and development."""

                async def get_capital_metrics(self):
                    """Get mock capital metrics."""
                    from datetime import datetime
                    from decimal import Decimal

                    from src.core.types import CapitalMetrics

                    return CapitalMetrics(
                        total_capital=Decimal("10000.00"),
                        allocated_capital=Decimal("5000.00"),
                        utilized_capital=Decimal("2500.00"),
                        available_capital=Decimal("5000.00"),
                        allocation_ratio=Decimal("0.5"),
                        utilization_ratio=Decimal("0.5"),
                        currency="USD",
                        last_updated=datetime.utcnow(),
                    )

                async def allocate_capital(self, **kwargs):
                    return {"allocation_id": "mock_alloc_123", "status": "allocated"}

                async def release_capital(self, **kwargs):
                    return True

                async def update_utilization(self, **kwargs):
                    return True

                async def get_all_allocations(self):
                    return []

                async def get_allocations_by_strategy(self, strategy_id):
                    return []

            capital_service = MockCapitalService()

        from src.web_interface.services.capital_service import WebCapitalService

        return WebCapitalService(
            capital_service=capital_service,
            currency_service=currency_service,
            fund_flow_service=fund_flow_service,
        )

    injector.register_factory("WebCapitalService", _create_web_capital_service, singleton=True)


def _register_utility_services(
    injector: DependencyInjector, factory: "WebInterfaceFactory"
) -> None:
    """Register utility services."""

    def _create_web_data_service():
        """Factory function for WebDataService with DI."""
        from src.web_interface.services.data_service import WebDataService

        data_service = (
            injector.resolve("DataService") if injector.has_service("DataService") else None
        )
        return WebDataService(data_service=data_service)

    def _create_web_exchange_service():
        """Factory function for WebExchangeService with DI."""
        from src.web_interface.services.exchange_service import WebExchangeService

        exchange_service = (
            injector.resolve("ExchangeService") if injector.has_service("ExchangeService") else None
        )
        return WebExchangeService(exchange_service=exchange_service)

    injector.register_factory("WebDataService", _create_web_data_service, singleton=True)
    injector.register_factory("WebExchangeService", _create_web_exchange_service, singleton=True)


# Service locator functions following factory pattern for decoupled service access
def get_web_portfolio_service(injector: DependencyInjector = None) -> "WebPortfolioService":
    """Service locator for web portfolio service with factory pattern."""
    if injector is None:
        from src.core.dependency_injection import get_global_injector

        injector = get_global_injector()

    # Ensure services are registered
    if not injector.has_service("WebPortfolioService"):
        register_web_interface_services(injector)

    return injector.resolve("WebPortfolioService")


def get_web_trading_service(injector: DependencyInjector = None) -> "WebTradingService":
    """Service locator for web trading service with factory pattern."""
    if injector is None:
        from src.core.dependency_injection import get_global_injector

        injector = get_global_injector()

    if not injector.has_service("WebTradingService"):
        register_web_interface_services(injector)

    return injector.resolve("WebTradingService")


def get_web_bot_service(injector: DependencyInjector = None) -> "WebBotService":
    """Service locator for web bot service with factory pattern."""
    if injector is None:
        from src.core.dependency_injection import get_global_injector

        injector = get_global_injector()

    if not injector.has_service("WebBotService"):
        register_web_interface_services(injector)

    return injector.resolve("WebBotService")


def get_web_monitoring_service(injector: DependencyInjector = None) -> "WebMonitoringService":
    """Service locator for web monitoring service with factory pattern."""
    if injector is None:
        from src.core.dependency_injection import get_global_injector

        injector = get_global_injector()

    if not injector.has_service("WebMonitoringService"):
        register_web_interface_services(injector)

    return injector.resolve("WebMonitoringService")


def get_web_risk_service(injector: DependencyInjector = None) -> "WebRiskService":
    """Service locator for web risk service with factory pattern."""
    if injector is None:
        from src.core.dependency_injection import get_global_injector

        injector = get_global_injector()

    if not injector.has_service("WebRiskService"):
        register_web_interface_services(injector)

    return injector.resolve("WebRiskService")


def get_api_facade_service(injector: DependencyInjector = None):
    """Service locator for API facade with factory pattern."""
    if injector is None:
        from src.core.dependency_injection import get_global_injector

        injector = get_global_injector()

    if not injector.has_service("APIFacade"):
        register_web_interface_services(injector)

    return injector.resolve("APIFacade")


def get_web_interface_factory(injector: DependencyInjector = None) -> "WebInterfaceFactory":
    """Service locator for web interface factory."""
    if injector is None:
        from src.core.dependency_injection import get_global_injector

        injector = get_global_injector()

    if not injector.has_service("WebInterfaceFactory"):
        register_web_interface_services(injector)

    return injector.resolve("WebInterfaceFactory")
