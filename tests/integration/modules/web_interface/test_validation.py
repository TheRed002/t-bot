"""
Integration validation tests for web_interface module.

This test suite validates proper integration patterns:
1. Dependency injection works correctly
2. Service layers are properly used
3. Module boundaries are respected
4. Error handling is consistent
5. Async lifecycle is properly implemented
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.dependency_injection import DependencyInjector
from src.core.exceptions import ServiceError
from src.web_interface.di_registration import register_web_interface_services
from src.web_interface.facade.api_facade import APIFacade
from src.web_interface.factory import WebInterfaceFactory
from src.web_interface.services.analytics_service import WebAnalyticsService
from src.web_interface.services.trading_service import WebTradingService


class TestWebInterfaceDependencyInjection:
    """Test dependency injection patterns in web interface."""

    @pytest.fixture
    def injector(self):
        """Create a test dependency injector."""
        return DependencyInjector()

    @pytest.fixture
    def factory(self, injector):
        """Create a web interface factory with injector."""
        return WebInterfaceFactory(injector)

    def test_service_registration(self, injector):
        """Test that all web interface services register correctly."""
        # Register services
        register_web_interface_services(injector)

        # Verify core services are registered
        assert injector.has_service("WebInterfaceFactory")
        assert injector.has_service("APIFacade")
        assert injector.has_service("WebTradingService")
        assert injector.has_service("WebAnalyticsService")
        assert injector.has_service("ServiceRegistry")

    def test_factory_pattern_integration(self, factory):
        """Test factory pattern creates services with proper dependencies."""
        # Create API facade through factory
        api_facade = factory.create_api_facade()

        assert isinstance(api_facade, APIFacade)
        assert api_facade._service_registry is not None
        assert api_facade._injector is not None

    def test_service_dependency_injection(self, injector):
        """Test service dependency injection works correctly."""
        # Mock core services that web services depend on
        mock_analytics_service = Mock()
        injector.register_service("AnalyticsService", mock_analytics_service)

        # Register web interface services
        register_web_interface_services(injector)

        # Resolve web analytics service
        web_analytics = injector.resolve("WebAnalyticsService")

        # Verify it got the mocked dependency
        assert web_analytics.analytics_service is mock_analytics_service

    def test_service_locator_pattern(self, injector):
        """Test service locator functions work correctly."""
        from src.web_interface.di_registration import (
            get_api_facade_service,
            get_web_trading_service,
        )

        # Register services first
        register_web_interface_services(injector)

        # Test service locators
        trading_service = get_web_trading_service(injector)
        api_facade = get_api_facade_service(injector)

        assert trading_service is not None
        assert api_facade is not None

        # Verify same instance returned (singleton)
        assert get_web_trading_service(injector) is trading_service


class TestServiceLayerIntegration:
    """Test service layer integration patterns."""

    @pytest.fixture
    def trading_service(self):
        """Create trading service for testing."""
        return WebTradingService()

    @pytest.fixture
    def analytics_service(self):
        """Create analytics service with mock dependencies."""
        mock_analytics = Mock()
        return WebAnalyticsService(analytics_service=mock_analytics)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_service_initialization(self, trading_service):
        """Test service initialization patterns."""
        await trading_service.initialize()
        # Should not raise errors

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_service_cleanup(self, trading_service):
        """Test service cleanup patterns."""
        await trading_service.initialize()
        await trading_service.cleanup()
        # Should not raise errors

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_error_handling_consistency(self, analytics_service):
        """Test that services use consistent error handling."""
        # Mock the analytics service to raise an exception
        analytics_service.analytics_service.get_portfolio_metrics = Mock(
            side_effect=Exception("Test error")
        )

        with pytest.raises(ServiceError) as exc_info:
            await analytics_service.get_portfolio_metrics()

        assert "Failed to retrieve portfolio metrics" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_business_logic_in_service_layer(self, trading_service):
        """Test that business logic is properly in service layer."""
        # Test order validation (business logic should be in service)
        result = await trading_service.validate_order_request(
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        assert "valid" in result
        assert "validated_data" in result


class TestModuleBoundaryValidation:
    """Test module boundary adherence."""

    def test_no_direct_repository_imports(self):
        """Test that web interface doesn't import repositories directly."""
        import src.web_interface.services.analytics_service as analytics_module
        import src.web_interface.services.trading_service as trading_module

        # Check module source for direct repository imports
        analytics_source = analytics_module.__file__
        trading_source = trading_module.__file__

        with open(analytics_source) as f:
            analytics_content = f.read()
        with open(trading_source) as f:
            trading_content = f.read()

        # Should not have direct repository imports
        assert "from src.database.repositories" not in analytics_content
        assert "from src.database.repositories" not in trading_content

    def test_proper_exception_usage(self):
        """Test that services use proper core exceptions."""
        import src.web_interface.services.analytics_service as analytics_module

        with open(analytics_module.__file__) as f:
            content = f.read()

        # Should import from core.exceptions
        assert "from src.core.exceptions import" in content

    def test_service_interface_compliance(self):
        """Test that services implement proper interfaces."""
        from src.web_interface.interfaces import WebTradingServiceInterface

        trading_service = WebTradingService()
        assert isinstance(trading_service, WebTradingServiceInterface)


class TestAPIFacadeIntegration:
    """Test API facade integration patterns."""

    @pytest.fixture
    def api_facade(self):
        """Create API facade with mock services."""
        mock_trading = Mock()
        mock_bot = Mock()
        return APIFacade(trading_service=mock_trading, bot_service=mock_bot)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_facade_initialization(self, api_facade):
        """Test facade properly initializes services."""
        # Mock service initialization
        api_facade._trading_service.initialize = AsyncMock()
        api_facade._bot_service.initialize = AsyncMock()

        await api_facade.initialize()

        assert api_facade._initialized
        api_facade._trading_service.initialize.assert_called_once()
        api_facade._bot_service.initialize.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_facade_dependency_configuration(self):
        """Test facade can configure dependencies from injector."""
        injector = DependencyInjector()
        mock_trading = Mock()
        injector.register_service("WebTradingService", mock_trading)

        facade = APIFacade()
        facade.configure_dependencies(injector)

        assert facade._trading_service is mock_trading

    def test_facade_service_availability_checks(self, api_facade):
        """Test facade properly checks service availability."""
        # Remove trading service
        api_facade._trading_service = None

        with pytest.raises(ValueError, match="Trading service not available"):
            # This should be a sync method that checks availability
            api_facade.place_order.__code__  # Just accessing to trigger the check in real usage


class TestAsyncLifecycleIntegration:
    """Test async lifecycle integration patterns."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_service_lifecycle_coordination(self):
        """Test that services coordinate lifecycle properly."""
        injector = DependencyInjector()
        factory = WebInterfaceFactory(injector)

        # Create facade
        api_facade = factory.create_api_facade()

        # Test initialization
        await api_facade.initialize()
        assert api_facade._initialized

        # Test cleanup
        await api_facade.cleanup()
        assert not api_facade._initialized


class TestErrorPropagationIntegration:
    """Test error propagation across service boundaries."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_service_error_propagation(self):
        """Test errors propagate correctly between service layers."""
        # Create service with failing dependency
        mock_analytics = Mock()
        mock_analytics.get_portfolio_metrics = Mock(side_effect=Exception("Database error"))

        service = WebAnalyticsService(analytics_service=mock_analytics)

        # Should convert to ServiceError
        with pytest.raises(ServiceError):
            await service.get_portfolio_metrics()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_validation_error_propagation(self):
        """Test validation errors propagate correctly."""
        trading_service = WebTradingService()

        # Test with invalid data
        result = await trading_service.validate_order_request(
            symbol="",  # Invalid symbol
            side="buy",
            order_type="limit",
            quantity=Decimal("0"),  # Invalid quantity
            price=Decimal("50000.0"),
        )

        assert not result["valid"]
        assert len(result["errors"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
