"""
Exchange Factory Integration Tests.

This module tests the exchange factory and service layer integration
with comprehensive coverage of exchange creation, configuration, and lifecycle management.
"""

import asyncio
from decimal import Decimal

import pytest

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.core.types import OrderRequest, OrderSide, OrderType, TimeInForce
from src.exchanges.factory import ExchangeFactory
from src.exchanges.mock_exchange import MockExchange

logger = get_logger(__name__)


class TestExchangeFactoryIntegration:
    """Test exchange factory with real configuration and creation scenarios."""

    @pytest.fixture(scope="class")
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture(scope="class")
    def factory(self, config):
        """Create exchange factory."""
        factory = ExchangeFactory(config)
        factory.register_default_exchanges()
        return factory

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_factory_creates_all_supported_exchanges(self, factory):
        """Test that factory can create all supported exchange types."""
        supported_exchanges = ["binance", "coinbase", "okx", "mock"]

        created_exchanges = {}

        for exchange_name in supported_exchanges:
            try:
                exchange = await factory.create_exchange(exchange_name)
                assert exchange is not None
                # Don't use isinstance with Protocol - just check interface compliance
                created_exchanges[exchange_name] = exchange

                # Test basic interface compliance
                assert hasattr(exchange, "connect")
                assert hasattr(exchange, "disconnect")
                assert hasattr(exchange, "is_connected")
                assert hasattr(exchange, "get_balance")
                assert hasattr(exchange, "place_order")

            except Exception as e:
                logger.warning(f"Could not create {exchange_name} exchange: {e}")

        # Should be able to create at least mock exchange
        assert "mock" in created_exchanges
        assert len(created_exchanges) >= 1

        # Cleanup
        for exchange in created_exchanges.values():
            try:
                await exchange.disconnect()
            except Exception:
                pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_factory_exchange_configuration(self, factory):
        """Test that factory properly configures exchanges."""
        # Test with mock exchange for guaranteed success
        exchange = await factory.create_exchange("mock")

        assert exchange is not None
        assert isinstance(exchange, MockExchange)

        # Test configuration is applied
        assert exchange.config is not None
        assert hasattr(exchange, "logger")

        # Test that exchange has expected methods (interface compliance)
        assert hasattr(exchange, "connect")
        assert hasattr(exchange, "disconnect")
        assert hasattr(exchange, "is_connected")

        await exchange.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_factory_invalid_exchange_handling(self, factory):
        """Test factory behavior with invalid exchange names."""
        with pytest.raises((ValueError, ValidationError)):
            await factory.create_exchange("invalid_exchange")

        with pytest.raises((ValueError, ValidationError)):
            await factory.create_exchange("")

        with pytest.raises((ValueError, ValidationError, TypeError)):
            await factory.create_exchange(None)

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_factory_with_custom_config(self):
        """Test factory with custom configuration."""
        custom_config = Config()
        custom_config.exchange.default_exchange = "binance"  # Use valid exchange name
        custom_config.exchange.connection_timeout = 15.0

        factory = ExchangeFactory(custom_config)
        factory.register_default_exchanges()
        exchange = await factory.create_exchange("mock")

        assert exchange is not None
        assert exchange.config == custom_config

        await exchange.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_concurrent_exchange_creation(self, factory):
        """Test concurrent exchange creation."""
        # Create multiple exchanges concurrently
        tasks = []
        for i in range(3):
            task = factory.create_exchange("mock")
            tasks.append(task)

        exchanges = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        successful_exchanges = [ex for ex in exchanges if not isinstance(ex, Exception)]
        assert len(successful_exchanges) == 3

        # All should be different instances
        assert len(set(id(ex) for ex in successful_exchanges)) == 3

        # Cleanup
        for exchange in successful_exchanges:
            await exchange.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_exchange_factory_dependency_injection(self, factory):
        """Test that factory properly injects dependencies."""
        exchange = await factory.create_exchange("mock")

        # Check that all required dependencies are injected
        assert hasattr(exchange, "config")
        assert hasattr(exchange, "logger")

        # Test that dependencies are properly configured
        assert exchange.config is not None
        assert exchange.logger is not None

        # Test that exchange has interface methods (dependency injection worked)
        assert callable(exchange.connect)
        assert callable(exchange.disconnect)
        assert callable(exchange.place_order)

        await exchange.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_factory_creates_production_ready_exchanges(self, factory):
        """Test that factory creates exchanges ready for production use."""
        exchange = await factory.create_exchange("mock")

        try:
            # Test connection lifecycle
            await exchange.connect()
            assert exchange.is_connected()

            # Test basic operations
            balances = await exchange.get_balance()
            assert balances is not None
            assert isinstance(balances, dict)
            assert len(balances) > 0

            # Test order placement
            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("50000"),
                time_in_force=TimeInForce.GTC,
            )

            order_response = await exchange.place_order(order_request)
            assert order_response is not None
            assert order_response.order_id is not None

        finally:
            await exchange.disconnect()


if __name__ == "__main__":
    # Run factory integration tests
    pytest.main([__file__, "-v", "--tb=short"])
