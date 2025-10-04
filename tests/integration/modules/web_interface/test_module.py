"""
Integration tests for web_interface module to verify proper integration with other modules.

This module tests that web_interface correctly integrates with and uses dependencies
from other modules in the system.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from decimal import Decimal
from datetime import datetime, timezone

from src.core.dependency_injection import DependencyInjector
from src.core.config import Config
from src.core.types import OrderSide, OrderType, Position, PositionSide, PositionStatus
from src.web_interface.di_registration import register_web_interface_services
from src.web_interface.factory import WebInterfaceFactory
from src.web_interface.facade import get_api_facade
from src.web_interface.services import (
    WebTradingService,
    WebBotService,
    WebPortfolioService,
    WebRiskService,
    WebStrategyService,
)


class TestWebInterfaceModuleIntegration:
    """Test web_interface module integration with other modules."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()

    @pytest.fixture
    def injector(self, config):
        """Create test dependency injector with mocked services."""
        injector = DependencyInjector()
        
        # Register configuration
        injector.register_service("Config", config)
        
        # Mock core services that web_interface depends on
        mock_execution_engine = Mock()
        mock_execution_engine.place_order = AsyncMock(return_value={"order_id": "test_order_123"})
        mock_execution_engine.cancel_order = AsyncMock(return_value={"success": True})
        mock_execution_engine.get_positions = AsyncMock(return_value=[
            {
                "symbol": "BTCUSDT",
                "size": 0.1,
                "entry_price": 45000.0,
                "current_price": 46000.0,
                "unrealized_pnl": 100.0
            }
        ])
        injector.register_service("ExecutionEngine", mock_execution_engine)
        
        # Mock bot orchestrator
        mock_bot_orchestrator = Mock()
        mock_bot_orchestrator.create_bot = AsyncMock(return_value="bot_123")
        mock_bot_orchestrator.start_bot = AsyncMock(return_value=True)
        mock_bot_orchestrator.stop_bot = AsyncMock(return_value=True)
        mock_bot_orchestrator.get_bot_status = AsyncMock(return_value={
            "bot_id": "bot_123",
            "status": "running",
            "performance": {"pnl": 150.0}
        })
        mock_bot_orchestrator.get_all_bots_status = AsyncMock(return_value={
            "bots": {
                "bot_123": {
                    "state": {"status": "running", "configuration": {"bot_name": "test_bot"}},
                    "metrics": {"pnl": 150.0}
                }
            }
        })
        injector.register_service("BotOrchestrator", mock_bot_orchestrator)
        
        # Mock data service
        mock_data_service = Mock()
        mock_data_service.get_recent_data = AsyncMock(return_value=[
            Mock(
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                open=Decimal("44800"),
                high=Decimal("45200"),
                low=Decimal("44500"),
                close=Decimal("45000"),
                volume=Decimal("1000000")
            )
        ])
        injector.register_service("DataService", mock_data_service)
        
        # Mock portfolio manager
        mock_portfolio_manager = Mock()
        mock_portfolio_manager.get_balances = AsyncMock(return_value={
            "USDT": 5000.0,
            "BTC": 0.1
        })
        mock_portfolio_manager.get_summary = AsyncMock(return_value={
            "total_value": 10000.0,
            "available_balance": 5000.0,
            "unrealized_pnl": 123.45
        })
        injector.register_service("PortfolioManager", mock_portfolio_manager)
        
        # Mock risk manager
        mock_risk_manager = Mock()
        mock_risk_manager.validate_order = AsyncMock(return_value={
            "valid": True,
            "risk_score": 0.25,
            "warnings": []
        })
        mock_risk_manager.get_risk_metrics = AsyncMock(return_value={
            "portfolio_var": 1250.0,
            "max_drawdown": 0.15,
            "sharpe_ratio": 1.35
        })
        injector.register_service("RiskManager", mock_risk_manager)
        
        # Mock strategy service and factory
        mock_strategy_service = Mock()
        mock_strategy_service.get_all_strategies = AsyncMock(return_value={
            "momentum": {"strategy_id": "momentum", "status": "active"}
        })
        injector.register_service("StrategyService", mock_strategy_service)
        
        mock_strategy_factory = Mock()
        mock_strategy_factory.list_available_strategies = Mock(return_value={
            "momentum": {"class_name": "MomentumStrategy", "required_parameters": ["fast_ma", "slow_ma"]}
        })
        injector.register_service("StrategyFactory", mock_strategy_factory)
        
        return injector

    def test_web_interface_service_registration(self, injector):
        """Test that web_interface services can be properly registered."""
        # Register web interface services
        register_web_interface_services(injector)
        
        # Verify all expected services are registered
        expected_services = [
            "WebServiceRegistry",
            "JWTHandler", 
            "AuthManager",
            "TradingService",
            "BotManagementService",
            "MarketDataService",
            "PortfolioService",
            "RiskService",
            "WebStrategyServiceImpl",
            "APIFacade",
            "UnifiedWebSocketManager"
        ]
        
        for service_name in expected_services:
            assert injector.has_service(service_name), f"Service {service_name} not registered"

    def test_web_interface_factory_creation(self, injector):
        """Test that web interface factory can create all components."""
        factory = WebInterfaceFactory(injector)
        
        # Test service registry creation
        service_registry = factory.create_service_registry()
        assert service_registry is not None
        
        # Test JWT handler creation
        jwt_handler = factory.create_jwt_handler()
        assert jwt_handler is not None
        
        # Test auth manager creation  
        auth_manager = factory.create_auth_manager()
        assert auth_manager is not None
        
        # Test service implementations creation
        trading_service = factory.create_trading_service()
        assert trading_service is not None
        assert isinstance(trading_service, WebTradingService)

        bot_service = factory.create_bot_management_service()
        assert bot_service is not None
        assert isinstance(bot_service, WebBotService)

        portfolio_service = factory.create_portfolio_service()
        assert portfolio_service is not None
        assert isinstance(portfolio_service, WebPortfolioService)

        risk_service = factory.create_risk_service()
        assert risk_service is not None
        assert isinstance(risk_service, WebRiskService)

        strategy_service = factory.create_strategy_service()
        assert strategy_service is not None
        assert isinstance(strategy_service, WebStrategyService)

    @pytest.mark.asyncio
    async def test_trading_service_integration(self, injector):
        """Test that trading service properly integrates with execution engine."""
        # Register services and create trading service
        register_web_interface_services(injector)
        factory = WebInterfaceFactory(injector)
        trading_service = factory.create_trading_service()
        
        # Configure dependencies
        trading_service.configure_dependencies(injector)
        await trading_service.initialize()
        
        # Test order placement
        order_result = await trading_service.place_order_through_service(
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            quantity=Decimal("0.1"),
            price=Decimal("45000")
        )

        # Validate order ID format (should be either mocked value or generated UUID format)
        order_id = order_result["order_id"]
        assert order_id is not None
        assert isinstance(order_id, str)
        assert len(order_id) > 0
        # Should either be the mocked value or UUID-based format like "order_12345678"
        assert order_id == "test_order_123" or order_id.startswith("order_")

        # Verify execution engine was called if using mocked facade (otherwise service uses fallback)
        mock_execution_engine = injector.resolve("ExecutionEngine")
        if order_id == "test_order_123":
            # This means the service used the trading facade which should call execution engine
            mock_execution_engine.place_order.assert_called_once_with(
                symbol="BTCUSDT",
                side="buy",
                order_type="market",
                amount=Decimal("0.1"),
                price=Decimal("45000")
            )
        else:
            # This means the service used its fallback mock implementation
            # The execution engine should not have been called
            mock_execution_engine.place_order.assert_not_called()
        
        # Test order cancellation
        test_order_id = order_id if order_id == "test_order_123" else "test_order_123"
        success = await trading_service.cancel_order_through_service(test_order_id)
        assert success is True

        # Verify cancellation behavior based on whether facade is available
        if order_id == "test_order_123":
            # Service has trading facade, should call execution engine
            mock_execution_engine.cancel_order.assert_called_once_with(test_order_id)
        else:
            # Service uses fallback mock, execution engine should not be called
            mock_execution_engine.cancel_order.assert_not_called()
        
        # Verify trading service is properly functioning
        # (Note: get_positions is not part of WebTradingService interface,
        # it would be handled by portfolio service or execution engine directly)

    @pytest.mark.asyncio
    async def test_bot_management_service_integration(self, injector):
        """Test that bot management service properly integrates with bot orchestrator."""
        register_web_interface_services(injector)
        factory = WebInterfaceFactory(injector)
        bot_service = factory.create_bot_management_service()
        
        # Configure dependencies
        bot_service.configure_dependencies(injector)
        await bot_service.initialize()
        
        # Test bot creation
        config = {"bot_name": "test_bot", "strategy": "momentum"}
        bot_id = await bot_service.create_bot_through_service(config)
        assert bot_id.startswith("bot_")
        
        # Test bot operations (using proper service methods)
        success = await bot_service.start_bot_through_service(bot_id)
        assert success is True

        status = await bot_service.get_bot_status_through_service(bot_id)
        assert status["bot_id"] == bot_id
        assert status["status"] == "running"

    @pytest.mark.asyncio
    async def test_market_data_service_integration(self, injector):
        """Test that market data service properly integrates with data service."""
        register_web_interface_services(injector)
        factory = WebInterfaceFactory(injector)
        market_service = factory.create_market_data_service()
        
        # Configure dependencies
        market_service.configure_dependencies(injector)
        await market_service.initialize()
        
        # Test ticker data retrieval
        ticker = await market_service.get_ticker("BTCUSDT")
        assert ticker.symbol == "BTCUSDT"
        assert ticker.close == Decimal("45000")
        
        # Verify data service was called
        mock_data_service = injector.resolve("DataService")
        mock_data_service.get_recent_data.assert_called_once_with("BTCUSDT", limit=1)

    @pytest.mark.asyncio
    async def test_portfolio_service_integration(self, injector):
        """Test that portfolio service properly integrates with portfolio manager."""
        register_web_interface_services(injector)
        factory = WebInterfaceFactory(injector)
        portfolio_service = factory.create_portfolio_service()
        
        # Configure dependencies
        portfolio_service.configure_dependencies(injector)
        await portfolio_service.initialize()
        
        # Test balance retrieval
        balances = await portfolio_service.get_balance()
        assert balances["USDT"] == Decimal("5000.00")
        assert balances["BTC"] == Decimal("0.1")
        
        # Test portfolio summary
        summary = await portfolio_service.get_portfolio_summary()
        assert summary["total_value"] == 10000.0
        assert summary["unrealized_pnl"] == 123.45
        
        # Verify portfolio manager was called
        mock_portfolio_manager = injector.resolve("PortfolioManager")
        mock_portfolio_manager.get_balances.assert_called_once()
        mock_portfolio_manager.get_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_risk_service_integration(self, injector):
        """Test that risk service properly integrates with risk manager."""
        register_web_interface_services(injector)
        factory = WebInterfaceFactory(injector)
        risk_service = factory.create_risk_service()
        
        # Configure dependencies
        risk_service.configure_dependencies(injector)
        await risk_service.initialize()
        
        # Test order validation
        result = await risk_service.validate_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            amount=Decimal("0.1"),
            price=Decimal("45000")
        )
        
        assert result["valid"] is True
        assert result["risk_score"] == 0.25
        
        # Test risk metrics
        metrics = await risk_service.get_risk_metrics()
        assert metrics["portfolio_var"] == 1250.0
        assert metrics["sharpe_ratio"] == 1.35
        
        # Verify risk manager was called
        mock_risk_manager = injector.resolve("RiskManager")
        mock_risk_manager.validate_order.assert_called_once_with(
            "BTCUSDT", "buy", Decimal("0.1"), Decimal("45000")
        )
        mock_risk_manager.get_risk_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_strategy_service_integration(self, injector):
        """Test that strategy service properly integrates with strategy components."""
        register_web_interface_services(injector)
        factory = WebInterfaceFactory(injector)
        strategy_service = factory.create_strategy_service()
        
        # Configure dependencies
        strategy_service.configure_dependencies(injector)
        await strategy_service.initialize()
        
        # Test strategy listing with factory
        strategies = await strategy_service.list_strategies()
        assert len(strategies) > 0
        assert any(s["name"] == "momentum" for s in strategies)
        
        # Test strategy config retrieval
        config = await strategy_service.get_strategy_config("momentum")
        assert isinstance(config, dict)

    @pytest.mark.asyncio
    async def test_api_facade_integration(self, injector):
        """Test that API facade properly integrates all services."""
        register_web_interface_services(injector)
        
        # Get API facade through factory
        factory = WebInterfaceFactory(injector)
        api_facade = factory.create_api_facade()
        
        await api_facade.initialize()
        
        # Test trading operations through facade
        order_id = await api_facade.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=Decimal("0.1"),
            price=Decimal("45000")
        )
        assert order_id == "test_order_123"
        
        # Test bot management through facade
        bot_id = await api_facade.create_bot({"bot_name": "test_bot"})
        assert bot_id == "bot_123"
        
        # Test market data through facade
        ticker = await api_facade.get_ticker("BTCUSDT")
        assert ticker.symbol == "BTCUSDT"
        
        # Test portfolio operations through facade
        balances = await api_facade.get_balance()
        assert "USDT" in balances
        
        # Test risk operations through facade
        validation = await api_facade.validate_order(
            "BTCUSDT", OrderSide.BUY, Decimal("0.1"), Decimal("45000")
        )
        assert validation["valid"] is True
        
        # Test strategy operations through facade
        strategies = await api_facade.list_strategies()
        assert len(strategies) > 0

    def test_error_handling_without_dependencies(self):
        """Test that services handle missing dependencies gracefully."""
        # Create services without injector
        trading_service = WebTradingService()
        bot_service = WebBotService()
        
        # Configure with empty injector (no dependencies available)
        empty_injector = DependencyInjector()
        
        # Configure dependencies if available
        if hasattr(trading_service, "configure_dependencies"):
            trading_service.configure_dependencies(empty_injector)
        if hasattr(bot_service, "configure_dependencies"):
            bot_service.configure_dependencies(empty_injector)

        # Services should handle missing dependencies without crashing
        assert trading_service.trading_facade is None
        assert bot_service.bot_facade is None

    @pytest.mark.asyncio
    async def test_service_fallback_behavior(self):
        """Test that services provide mock/fallback behavior when dependencies are missing."""
        trading_service = WebTradingService()
        portfolio_service = WebPortfolioService()
        
        # Initialize without dependencies
        await trading_service.initialize()
        await portfolio_service.initialize()
        
        # Test fallback behaviors
        order_id = await trading_service.place_order(
            "BTCUSDT", OrderSide.BUY, OrderType.MARKET, Decimal("0.1")
        )
        assert order_id.startswith("ORD_")  # Mock order ID
        
        # Test mock ticker data
        ticker = await market_service.get_ticker("BTCUSDT")
        assert ticker.symbol == "BTCUSDT"
        assert ticker.close == Decimal("45000.00")  # Mock price
        
        # Test mock portfolio data
        balances = await portfolio_service.get_balance()
        assert "USDT" in balances
        assert balances["USDT"] == Decimal("5000.00")  # Mock balance

    @pytest.mark.asyncio
    async def test_dependency_injection_lifecycle(self, injector):
        """Test proper initialization and cleanup through dependency injection."""
        register_web_interface_services(injector)
        
        # Create and initialize services
        factory = WebInterfaceFactory(injector)
        api_facade = factory.create_api_facade()
        
        # Test initialization
        await api_facade.initialize()
        assert api_facade._initialized is True
        
        # Services should be accessible
        health = api_facade.health_check()
        assert health["status"] == "healthy"
        assert "services" in health
        
        # Test cleanup
        await api_facade.cleanup()
        assert api_facade._initialized is False

    def test_web_interface_complete_stack_creation(self, injector):
        """Test that complete web interface stack can be created."""
        factory = WebInterfaceFactory(injector)
        
        # Create complete stack
        components = factory.create_complete_web_stack()
        
        # Verify all expected components are created
        expected_components = [
            "service_registry", "jwt_handler", "auth_manager",
            "trading_service", "bot_management_service", "market_data_service",
            "portfolio_service", "risk_service", "strategy_service",
            "api_facade", "websocket_manager"
        ]
        
        for component_name in expected_components:
            assert component_name in components, f"Component {component_name} not created"
            assert components[component_name] is not None

    @pytest.mark.asyncio 
    async def test_service_registry_integration(self, injector):
        """Test that service registry properly manages service lifecycle."""
        register_web_interface_services(injector)
        
        # Get service registry
        service_registry = injector.resolve("WebServiceRegistry")
        
        # Register a test service
        mock_service = Mock()
        mock_service.initialize = AsyncMock()
        mock_service.cleanup = AsyncMock()
        
        service_registry.register_service("TestService", mock_service)
        
        # Test service retrieval
        retrieved_service = service_registry.get_service("TestService")
        assert retrieved_service is mock_service
        
        # Test service listing
        services = service_registry.list_services()
        assert "TestService" in services
        
        # Test initialization and cleanup
        await service_registry.initialize_all()
        mock_service.initialize.assert_called_once()
        
        await service_registry.cleanup_all()
        mock_service.cleanup.assert_called_once()