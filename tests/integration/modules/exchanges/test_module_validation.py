"""
Integration test validation for exchanges module.

This test validates that the exchanges module properly integrates with other modules:
- Dependency injection patterns
- Service consumption by other modules
- Error handling propagation
- Data contract compliance
- Architecture layer compliance
"""

import asyncio
import pytest
import pytest_asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import Config
from src.core.dependency_injection import DependencyContainer
from src.core.exceptions import ServiceError, ValidationError
from src.core.types import OrderRequest, OrderResponse, OrderStatus, OrderSide, OrderType
from src.exchanges.di_registration import register_exchange_dependencies
from src.exchanges.factory import ExchangeFactory
from src.exchanges.service import ExchangeService
from src.exchanges.base import BaseExchange


class MockExchange(BaseExchange):
    """Mock exchange for testing integration."""
    
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self._connected = True
        
    async def connect(self) -> bool:
        self._connected = True
        return True
        
    async def disconnect(self) -> None:
        self._connected = False
        
    async def health_check(self) -> bool:
        return self._connected
        
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        return OrderResponse(
            order_id="test_order_123",
            exchange_order_id="exchange_123",
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            status=OrderStatus.FILLED
        )
        
    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        return OrderResponse(
            order_id=order_id,
            exchange_order_id="exchange_123",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("100.0"),
            status=OrderStatus.CANCELLED
        )
        
    async def get_order_status(self, symbol: str, order_id: str) -> OrderResponse:
        return OrderResponse(
            order_id=order_id,
            exchange_order_id="exchange_123",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("100.0"),
            status=OrderStatus.FILLED
        )


@pytest.fixture
def config():
    """Test configuration."""
    config = Config()
    config.environment = "test"
    return config


@pytest_asyncio.fixture
async def exchange_factory(config, container):
    """Exchange factory instance."""
    from src.exchanges.base import BaseMockExchange

    factory = ExchangeFactory(config, container)
    factory.register_exchange("test", BaseMockExchange)
    return factory


@pytest.fixture
def exchange_service(exchange_factory, config):
    """Exchange service instance."""
    return ExchangeService(exchange_factory, config)


class TestExchangesModuleIntegration:
    """Test exchanges module integration patterns."""
    
    def test_dependency_injection_registration(self, config):
        """Test that DI registration works properly."""
        container = DependencyContainer()
        
        # Test registration doesn't raise exceptions
        register_exchange_dependencies(container, config)

        # Verify services are registered (DependencyContainer uses 'has', not 'is_registered')
        assert container.has("exchange_factory")
        assert container.has("exchange_service")
        
        # Test service creation
        factory = container.get("exchange_factory")
        assert factory is not None
        assert isinstance(factory, ExchangeFactory)
        
        service = container.get("exchange_service")
        assert service is not None
        assert isinstance(service, ExchangeService)
    
    def test_factory_interface_compliance(self, exchange_factory):
        """Test that factory implements required interface methods."""
        # Test interface methods exist and work
        supported = exchange_factory.get_supported_exchanges()
        assert isinstance(supported, list)
        
        available = exchange_factory.get_available_exchanges()
        assert isinstance(available, list)
        
        assert exchange_factory.is_exchange_supported("test")
        assert not exchange_factory.is_exchange_supported("nonexistent")
    
    @pytest.mark.asyncio
    async def test_factory_exchange_creation(self, exchange_factory):
        """Test factory creates exchanges properly."""
        from src.exchanges.base import BaseMockExchange

        # Test successful creation
        exchange = await exchange_factory.create_exchange("test")
        assert exchange is not None
        assert isinstance(exchange, BaseMockExchange)
        
        # Test interface method
        exchange_via_get = await exchange_factory.get_exchange("test")
        assert exchange_via_get is not None
        
        # Test validation error for unsupported exchange
        with pytest.raises(ValidationError):
            await exchange_factory.create_exchange("nonexistent")
    
    @pytest.mark.asyncio
    async def test_service_layer_patterns(self, exchange_service):
        """Test service layer properly uses factory."""
        await exchange_service.start()
        
        try:
            # Test service methods work through proper abstraction
            exchanges = exchange_service.get_available_exchanges()
            assert isinstance(exchanges, list)
            
            # Test service provides business logic layer
            status = await exchange_service.get_service_health()
            assert "service" in status
            assert "status" in status
            
        finally:
            await exchange_service.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, exchange_service):
        """Test error handling propagates properly across module boundaries."""
        await exchange_service.start()

        try:
            # Test validation errors are properly propagated
            with pytest.raises(ValidationError):
                await exchange_service.get_exchange("")  # Empty name should fail

            # Test validation errors for unsupported exchanges
            with pytest.raises(ValidationError, match="not supported"):
                await exchange_service.get_exchange("nonexistent_exchange")

        finally:
            await exchange_service.stop()
    
    @pytest.mark.asyncio
    async def test_architecture_layer_compliance(self, exchange_service, exchange_factory):
        """Test that service layer doesn't bypass architecture patterns."""
        # Mock factory to test service dependency
        mock_factory = AsyncMock()
        mock_factory.get_supported_exchanges.return_value = ["test"]
        mock_factory.get_available_exchanges.return_value = ["test"]
        
        # Create service with mocked factory
        test_config = Config()
        service = ExchangeService(mock_factory, test_config)
        
        # Test service calls factory methods (not direct exchange access)
        supported = service.get_supported_exchanges()
        mock_factory.get_supported_exchanges.assert_called_once()
        
        available = service.get_available_exchanges()
        mock_factory.get_available_exchanges.assert_called_once()

    # DELETED: test_database_service_integration - violated NO MOCKS policy (used AsyncMock)
    # Real database integration is tested in other integration tests

    @pytest.mark.asyncio
    async def test_web_interface_service_consumption(self, exchange_service):
        """Test that web interface properly consumes exchange service."""
        # Import web exchange service
        from src.web_interface.services.exchange_service import WebExchangeService
        
        # Create web service with exchange service dependency
        web_service = WebExchangeService(exchange_service=exchange_service)
        
        # Verify web service uses exchange service properly
        assert web_service.exchange_service is not None
        assert web_service.exchange_service == exchange_service
    
    @pytest.mark.asyncio  
    async def test_service_health_checks(self, exchange_service):
        """Test that health checks work across service boundaries."""
        await exchange_service.start()
        
        try:
            # Test service health check
            health = await exchange_service.get_service_health()
            
            assert "service" in health
            assert "status" in health
            assert "active_exchanges" in health
            assert isinstance(health["active_exchanges"], int)
            
        finally:
            await exchange_service.stop()


class TestDataContractCompliance:
    """Test data contracts are properly maintained across module boundaries."""
    
    @pytest.mark.asyncio
    async def test_order_response_contract(self, exchange_factory, container):
        """Test OrderResponse contract is maintained."""
        exchange = await exchange_factory.create_exchange("test")
        exchange.configure_dependencies(container)
        await exchange.start()

        try:
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("50000.0")
            )

            response = await exchange.place_order(order_request)

            # Verify response has all required fields
            assert hasattr(response, "order_id")
            assert hasattr(response, "symbol")
            assert hasattr(response, "side")
            assert hasattr(response, "order_type")
            assert hasattr(response, "quantity")
            assert hasattr(response, "price")
            assert hasattr(response, "status")

            # Verify types are correct
            assert isinstance(response.quantity, Decimal)
            assert isinstance(response.price, Decimal)
            assert isinstance(response.status, OrderStatus)

        finally:
            await exchange.stop()

    @pytest.mark.asyncio
    async def test_interface_method_signatures(self, exchange_factory, container):
        """Test that interface method signatures are consistent."""
        exchange = await exchange_factory.create_exchange("test")
        exchange.configure_dependencies(container)
        await exchange.start()

        try:
            # First place an order to have something to query
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("50000.0")
            )
            place_response = await exchange.place_order(order_request)
            order_id = place_response.order_id

            # Test get_order_status signature matches interface
            status_response = await exchange.get_order_status("BTCUSDT", order_id)
            assert isinstance(status_response, OrderResponse)

            # Test cancel_order signature matches interface
            cancel_response = await exchange.cancel_order("BTCUSDT", order_id)
            assert isinstance(cancel_response, OrderResponse)

        finally:
            await exchange.stop()


@pytest.mark.asyncio
async def test_full_integration_workflow():
    """Test complete integration workflow from DI to service usage."""
    from src.core.dependency_injection import DependencyInjector
    from src.core.di_master_registration import register_all_services

    config = Config()
    config.environment = "test"

    # Register all services using master registration
    container = register_all_services(injector=None, config=config)

    # Get services from container
    exchange_service = container.resolve("exchange_service")
    assert exchange_service is not None

    # Start service
    await exchange_service.start()

    try:
        # Test service functionality
        supported = exchange_service.get_supported_exchanges()
        assert isinstance(supported, list)

        health = await exchange_service.get_service_health()
        assert "service" in health

    finally:
        await exchange_service.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])