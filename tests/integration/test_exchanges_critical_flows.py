"""
Integration tests for critical exchange flows and cross-module interactions.

This test file targets additional coverage by testing integration scenarios
that span multiple exchange components and real-world workflows.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.core.config import Config
from src.core.exceptions import ServiceError, ValidationError
from src.core.types import OrderRequest, OrderResponse, OrderStatus
from src.exchanges.connection_manager import ConnectionManager
from src.exchanges.factory import ExchangeFactory
from src.exchanges.interfaces import IExchange
from src.exchanges.service import ExchangeService


@pytest.fixture
def mock_config():
    """Create mock config with exchange settings."""
    config = MagicMock(spec=Config)
    config.to_dict.return_value = {
        "exchanges": {
            "binance": {"api_key": "test_key", "api_secret": "test_secret"},
            "coinbase": {"api_key": "test_key", "api_secret": "test_secret"},
        }
    }
    config.exchange_service = MagicMock()
    config.exchange_service.default_timeout_seconds = 30
    config.exchange_service.max_retries = 3
    config.exchange_service.health_check_interval_seconds = 60
    return config


@pytest.fixture
def mock_rate_limiter():
    """Create mock rate limiter."""
    rate_limiter = AsyncMock()
    rate_limiter.acquire.return_value = None
    rate_limiter.can_proceed.return_value = True
    return rate_limiter


@pytest.fixture
def mock_connection_manager():
    """Create mock connection manager."""
    connection = AsyncMock(spec=ConnectionManager)
    return connection


class TestExchangeServiceFactoryIntegration:
    """Test integration between ExchangeService and ExchangeFactory."""

    async def test_service_factory_integration_success(self, mock_config):
        """Test successful integration between service and factory."""
        # Create mock factory
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_exchange = AsyncMock(spec=IExchange)
        mock_exchange.exchange_name = "binance"
        mock_exchange.health_check.return_value = True
        mock_factory.get_exchange.return_value = mock_exchange
        mock_factory.get_supported_exchanges.return_value = ["binance"]
        mock_factory.get_available_exchanges.return_value = ["binance"]

        # Create service
        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # Test exchange retrieval through service
        exchange = await service.get_exchange("binance")

        assert exchange == mock_exchange
        mock_factory.get_exchange.assert_called_once_with(
            exchange_name="binance", create_if_missing=True
        )

        await service.stop()

    async def test_service_factory_integration_factory_failure(self, mock_config):
        """Test service behavior when factory fails."""
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_factory.get_exchange.side_effect = Exception("Factory failed")
        mock_factory.get_supported_exchanges.return_value = ["binance"]

        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        with pytest.raises(ServiceError, match="Exchange retrieval failed"):
            await service.get_exchange("binance")

        await service.stop()


class TestExchangeOrderFlow:
    """Test complete order execution flows."""

    async def test_complete_order_placement_flow(self, mock_config):
        """Test complete order placement workflow."""
        # Setup mocks
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_exchange = AsyncMock(spec=IExchange)
        mock_exchange.exchange_name = "binance"
        mock_exchange.health_check.return_value = True

        # Setup order flow
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        order_response = OrderResponse(
            id="order-123",
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            status=OrderStatus.NEW,
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
            created_at=datetime.now(timezone.utc),
            exchange="binance",
        )

        mock_exchange.place_order.return_value = order_response
        mock_factory.get_exchange.return_value = mock_exchange
        mock_factory.get_supported_exchanges.return_value = ["binance"]

        # Create service and test flow
        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # Execute order placement
        result = await service.place_order("binance", order_request)

        assert result == order_response
        mock_exchange.place_order.assert_called_once_with(order_request)

        await service.stop()

    async def test_order_cancellation_flow(self, mock_config):
        """Test complete order cancellation workflow."""
        # Setup mocks
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_exchange = AsyncMock(spec=IExchange)
        mock_exchange.exchange_name = "binance"
        mock_exchange.health_check.return_value = True
        mock_exchange.cancel_order.return_value = True

        mock_factory.get_exchange.return_value = mock_exchange
        mock_factory.get_supported_exchanges.return_value = ["binance"]

        # Create service and test flow
        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # Execute cancellation
        result = await service.cancel_order("binance", "order-123")

        assert result is True
        mock_exchange.cancel_order.assert_called_once_with("", "order-123")

        await service.stop()

    async def test_order_status_tracking_flow(self, mock_config):
        """Test order status tracking workflow."""
        # Setup mocks
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_exchange = AsyncMock(spec=IExchange)
        mock_exchange.exchange_name = "binance"
        mock_exchange.health_check.return_value = True
        mock_exchange.get_order_status.return_value = OrderStatus.FILLED

        mock_factory.get_exchange.return_value = mock_exchange
        mock_factory.get_supported_exchanges.return_value = ["binance"]

        # Create service and test flow
        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # Execute status check
        status = await service.get_order_status("binance", "order-123")

        assert status == OrderStatus.FILLED
        mock_exchange.get_order_status.assert_called_once_with("", "order-123")

        await service.stop()


class TestMultiExchangeScenarios:
    """Test scenarios involving multiple exchanges."""

    async def test_multi_exchange_best_price_calculation(self, mock_config):
        """Test best price calculation across multiple exchanges."""
        # Setup mocks for multiple exchanges
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_factory.get_supported_exchanges.return_value = ["binance", "coinbase"]
        mock_factory.get_available_exchanges.return_value = ["binance", "coinbase"]

        # Create service
        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # Mock get_ticker_safe to return different prices
        async def mock_get_ticker_safe(exchange_name, symbol):
            if exchange_name == "binance":
                return MagicMock(ask_price=Decimal("50000"), bid_price=Decimal("49998"))
            elif exchange_name == "coinbase":
                return MagicMock(
                    ask_price=Decimal("49999"), bid_price=Decimal("49997")
                )  # Better buy price
            return None

        service._get_ticker_safe = mock_get_ticker_safe

        # Test best price for buying (should choose lower ask price)
        result = await service.get_best_price("BTCUSDT", "BUY", ["binance", "coinbase"])

        assert result["best_exchange"] == "coinbase"
        assert result["best_price"] == Decimal("49999")
        assert result["symbol"] == "BTCUSDT"
        assert result["side"] == "BUY"

        await service.stop()

    async def test_multi_exchange_with_failures(self, mock_config):
        """Test multi-exchange operations with some exchanges failing."""
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_factory.get_supported_exchanges.return_value = ["binance", "coinbase", "okx"]
        mock_factory.get_available_exchanges.return_value = ["binance", "coinbase", "okx"]

        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # Mock get_ticker_safe with mixed results
        async def mock_get_ticker_safe(exchange_name, symbol):
            if exchange_name == "binance":
                return MagicMock(bid_price=Decimal("50001"), ask_price=Decimal("50003"))
            elif exchange_name == "coinbase":
                return None  # Failed
            elif exchange_name == "okx":
                return MagicMock(
                    bid_price=Decimal("50002"), ask_price=Decimal("50004")
                )  # Higher bid for selling
            return None

        service._get_ticker_safe = mock_get_ticker_safe

        # Test best price for selling (should choose higher bid price)
        result = await service.get_best_price("BTCUSDT", "SELL", ["binance", "coinbase", "okx"])

        assert result["best_exchange"] == "okx"
        assert result["best_price"] == Decimal("50002")
        assert result["all_prices"]["binance"] == Decimal("50001")
        assert result["all_prices"]["coinbase"] is None  # Failed exchange
        assert result["all_prices"]["okx"] == Decimal("50002")

        await service.stop()


class TestExchangeHealthMonitoring:
    """Test health monitoring and failover scenarios."""

    async def test_exchange_health_monitoring_flow(self, mock_config):
        """Test health monitoring workflow."""
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_exchange = AsyncMock(spec=IExchange)
        mock_exchange.exchange_name = "binance"
        mock_exchange.health_check.return_value = True

        mock_factory.get_exchange.return_value = mock_exchange
        mock_factory.get_supported_exchanges.return_value = ["binance"]

        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # Add exchange to active pool
        await service.get_exchange("binance")

        # Test health status
        health = await service.get_service_health()

        assert health["service"] == "ExchangeService"
        assert health["active_exchanges"] == 1
        assert "binance" in health["exchanges"]
        assert health["exchanges"]["binance"]["healthy"] is True

        await service.stop()

    async def test_exchange_failover_scenario(self, mock_config):
        """Test exchange failover when primary becomes unhealthy."""
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_exchange = AsyncMock(spec=IExchange)
        mock_exchange.exchange_name = "binance"

        # First get_exchange: no health check (not cached yet)
        # Second get_exchange: health check returns False -> triggers failover
        # After recreation: health check could return True (but not tested here)
        health_responses = [False]  # unhealthy on first check, triggers failover
        mock_exchange.health_check.side_effect = health_responses

        mock_factory.get_exchange.return_value = mock_exchange
        mock_factory.get_supported_exchanges.return_value = ["binance"]

        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # First call - should work (exchange is created, no health check yet)
        exchange1 = await service.get_exchange("binance")
        assert exchange1 == mock_exchange
        assert service.exchange_factory.get_exchange.call_count == 1

        # Second call - should detect unhealthy exchange and recreate it
        exchange2 = await service.get_exchange("binance")
        assert exchange2 == mock_exchange
        assert service.exchange_factory.get_exchange.call_count == 2  # Called again due to failover
        mock_exchange.disconnect.assert_called_once()  # Unhealthy exchange was disconnected

        await service.stop()


class TestConcurrentOperations:
    """Test concurrent exchange operations."""

    async def test_concurrent_order_operations(self, mock_config):
        """Test concurrent order operations on the same exchange."""
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_exchange = AsyncMock(spec=IExchange)
        mock_exchange.exchange_name = "binance"
        mock_exchange.health_check.return_value = True

        # Setup concurrent order responses
        order_responses = [
            OrderResponse(id="order-1", symbol="BTCUSDT", side="buy", order_type="market", status=OrderStatus.NEW, quantity=Decimal("1.0"), created_at=datetime.now(timezone.utc), exchange="binance"),
            OrderResponse(id="order-2", symbol="ETHUSDT", side="sell", order_type="market", status=OrderStatus.NEW, quantity=Decimal("10.0"), created_at=datetime.now(timezone.utc), exchange="binance"),
            OrderResponse(id="order-3", symbol="ADAUSDT", side="buy", order_type="market", status=OrderStatus.NEW, quantity=Decimal("1000.0"), created_at=datetime.now(timezone.utc), exchange="binance"),
        ]

        mock_exchange.place_order.side_effect = order_responses
        mock_factory.get_exchange.return_value = mock_exchange
        mock_factory.get_supported_exchanges.return_value = ["binance"]

        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # Create concurrent orders
        orders = [
            OrderRequest(symbol="BTCUSDT", side="buy", order_type="market", quantity=Decimal("1.0")),
            OrderRequest(symbol="ETHUSDT", side="sell", order_type="market", quantity=Decimal("10.0")),
            OrderRequest(symbol="ADAUSDT", side="buy", order_type="market", quantity=Decimal("1000.0")),
        ]

        # Execute orders concurrently
        tasks = [service.place_order("binance", order) for order in orders]
        results = await asyncio.gather(*tasks)

        # Verify results
        assert len(results) == 3
        assert results[0].id == "order-1"
        assert results[1].id == "order-2"
        assert results[2].id == "order-3"

        # Verify all orders were placed
        assert mock_exchange.place_order.call_count == 3

        await service.stop()

    async def test_concurrent_multi_exchange_operations(self, mock_config):
        """Test concurrent operations across multiple exchanges."""
        # Setup multiple exchanges
        mock_factory = AsyncMock(spec=ExchangeFactory)

        binance_exchange = AsyncMock(spec=IExchange)
        binance_exchange.exchange_name = "binance"
        binance_exchange.health_check.return_value = True

        coinbase_exchange = AsyncMock(spec=IExchange)
        coinbase_exchange.exchange_name = "coinbase"
        coinbase_exchange.health_check.return_value = True

        # Factory returns different exchanges based on name
        def get_exchange_side_effect(exchange_name, **kwargs):
            if exchange_name == "binance":
                return binance_exchange
            elif exchange_name == "coinbase":
                return coinbase_exchange
            return None

        mock_factory.get_exchange.side_effect = get_exchange_side_effect
        mock_factory.get_supported_exchanges.return_value = ["binance", "coinbase"]

        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # Setup concurrent operations on different exchanges
        binance_order = OrderRequest(
            symbol="BTCUSDT", side="buy", order_type="market", quantity=Decimal("1.0")
        )
        coinbase_order = OrderRequest(
            symbol="BTCUSD", side="sell", order_type="market", quantity=Decimal("1.0")
        )

        binance_response = OrderResponse(
            id="binance-order-1", symbol="BTCUSDT", side="buy", order_type="market", status=OrderStatus.NEW, quantity=Decimal("1.0"), created_at=datetime.now(timezone.utc), exchange="binance"
        )
        coinbase_response = OrderResponse(
            id="coinbase-order-1", symbol="BTCUSD", side="sell", order_type="market", status=OrderStatus.NEW, quantity=Decimal("1.0"), created_at=datetime.now(timezone.utc), exchange="coinbase"
        )

        binance_exchange.place_order.return_value = binance_response
        coinbase_exchange.place_order.return_value = coinbase_response

        # Execute orders concurrently on different exchanges
        tasks = [
            service.place_order("binance", binance_order),
            service.place_order("coinbase", coinbase_order),
        ]

        results = await asyncio.gather(*tasks)

        # Verify results
        assert len(results) == 2
        assert results[0].id == "binance-order-1"
        assert results[1].id == "coinbase-order-1"

        # Verify orders were placed on correct exchanges
        binance_exchange.place_order.assert_called_once_with(binance_order)
        coinbase_exchange.place_order.assert_called_once_with(coinbase_order)

        await service.stop()


class TestErrorRecoveryFlows:
    """Test error recovery and resilience scenarios."""

    async def test_exchange_connection_recovery(self, mock_config):
        """Test exchange connection recovery after failure."""
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_exchange = AsyncMock(spec=IExchange)
        mock_exchange.exchange_name = "binance"

        # Simulate connection failures followed by recovery
        # First get_exchange: no health check (not cached)
        # Second get_exchange: health check returns False -> triggers recreation
        # Third get_exchange: health check returns True -> uses cached
        health_responses = [False, True]  # fail on first check -> healthy after recreation
        mock_exchange.health_check.side_effect = health_responses

        mock_factory.get_exchange.return_value = mock_exchange
        mock_factory.get_supported_exchanges.return_value = ["binance"]

        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # First call succeeds (no health check, exchange created)
        await service.get_exchange("binance")
        assert service.exchange_factory.get_exchange.call_count == 1

        # Second call detects unhealthy exchange and recreates
        await service.get_exchange("binance")
        assert service.exchange_factory.get_exchange.call_count == 2

        # Third call should use existing exchange if it's now healthy
        await service.get_exchange("binance")
        # If the recreated exchange is healthy, it should not call factory again
        assert service.exchange_factory.get_exchange.call_count == 2  # No additional call

        await service.stop()

    async def test_service_graceful_shutdown_with_errors(self, mock_config):
        """Test service graceful shutdown even when some operations fail."""
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_exchange1 = AsyncMock(spec=IExchange)
        mock_exchange1.exchange_name = "binance"
        mock_exchange1.health_check.return_value = True
        mock_exchange1.disconnect.side_effect = Exception("Disconnect failed")  # Simulate failure

        mock_exchange2 = AsyncMock(spec=IExchange)
        mock_exchange2.exchange_name = "coinbase"
        mock_exchange2.health_check.return_value = True
        mock_exchange2.disconnect.return_value = None  # Succeeds

        def get_exchange_side_effect(exchange_name, **kwargs):
            if exchange_name == "binance":
                return mock_exchange1
            elif exchange_name == "coinbase":
                return mock_exchange2
            return None

        mock_factory.get_exchange.side_effect = get_exchange_side_effect
        mock_factory.get_supported_exchanges.return_value = ["binance", "coinbase"]

        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # Add both exchanges to active pool
        await service.get_exchange("binance")
        await service.get_exchange("coinbase")

        assert len(service._active_exchanges) == 2

        # Stop service - should handle the disconnect failure gracefully
        await service.stop()

        # Service should still be stopped despite individual failures
        assert not service.is_running  # is_running is a property, not a method
        assert len(service._active_exchanges) == 0

        # Both exchanges should have had disconnect called
        mock_exchange1.disconnect.assert_called()
        mock_exchange2.disconnect.assert_called()


class TestWebSocketIntegration:
    """Test WebSocket integration scenarios."""

    async def test_websocket_subscription_flow(self, mock_config):
        """Test WebSocket subscription workflow."""
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_exchange = AsyncMock(spec=IExchange)
        mock_exchange.exchange_name = "binance"
        mock_exchange.health_check.return_value = True
        mock_exchange.subscribe_to_stream = AsyncMock()

        mock_factory.get_exchange.return_value = mock_exchange
        mock_factory.get_supported_exchanges.return_value = ["binance"]

        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # Test stream subscription
        callback = MagicMock()
        await service.subscribe_to_stream("binance", "BTCUSDT", callback)

        mock_exchange.subscribe_to_stream.assert_called_once_with("BTCUSDT", callback)

        await service.stop()

    async def test_websocket_subscription_failure(self, mock_config):
        """Test WebSocket subscription failure handling."""
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_exchange = AsyncMock(spec=IExchange)
        mock_exchange.exchange_name = "binance"
        mock_exchange.health_check.return_value = True
        mock_exchange.subscribe_to_stream.side_effect = Exception("WebSocket failed")

        mock_factory.get_exchange.return_value = mock_exchange
        mock_factory.get_supported_exchanges.return_value = ["binance"]

        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # Test stream subscription failure
        callback = MagicMock()
        with pytest.raises(ServiceError, match="Failed to subscribe to stream"):
            await service.subscribe_to_stream("binance", "BTCUSDT", callback)

        await service.stop()


class TestConfigurationScenarios:
    """Test various configuration scenarios."""

    async def test_service_with_minimal_config(self):
        """Test service with minimal configuration."""
        # Minimal config as dict
        config = {"basic": "config"}

        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_factory.get_supported_exchanges.return_value = ["binance"]

        service = ExchangeService(exchange_factory=mock_factory, config=config)

        # Should use default values
        assert service._default_timeout == 30
        assert service._max_retries == 3
        assert service._health_check_interval == 60

        await service.start()
        await service.stop()

    async def test_service_with_custom_config(self):
        """Test service with custom configuration."""
        config = MagicMock()
        config.to_dict.return_value = {"custom": "config"}
        config.exchange_service = MagicMock()
        config.exchange_service.default_timeout_seconds = 45
        config.exchange_service.max_retries = 5
        config.exchange_service.health_check_interval_seconds = 90

        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_factory.get_supported_exchanges.return_value = ["binance"]

        service = ExchangeService(exchange_factory=mock_factory, config=config)

        # Should use custom values
        assert service._default_timeout == 45
        assert service._max_retries == 5
        assert service._health_check_interval == 90

        await service.start()
        await service.stop()


class TestValidationIntegration:
    """Test integration with validation components."""

    async def test_order_validation_integration(self, mock_config):
        """Test integration between service and validation."""
        mock_factory = AsyncMock(spec=ExchangeFactory)
        mock_exchange = AsyncMock(spec=IExchange)
        mock_exchange.exchange_name = "binance"
        mock_exchange.health_check.return_value = True

        mock_factory.get_exchange.return_value = mock_exchange
        mock_factory.get_supported_exchanges.return_value = ["binance"]

        service = ExchangeService(exchange_factory=mock_factory, config=mock_config)

        await service.start()

        # Test validation of various order scenarios
        valid_order = OrderRequest(
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
        )

        # This should work
        await service._validate_order_request(valid_order)

        # Test invalid orders - Pydantic validation happens at construction
        # so we need to catch Pydantic's ValidationError
        with pytest.raises(PydanticValidationError):
            OrderRequest(
                symbol="",  # Empty symbol should fail
                side="buy",
                order_type="limit",
                quantity=Decimal("1.0"),
                price=Decimal("50000.0"),
            )

        with pytest.raises(PydanticValidationError):
            OrderRequest(
                symbol="BTCUSDT",
                side="",  # Empty side should fail
                order_type="limit",
                quantity=Decimal("1.0"),
                price=Decimal("50000.0"),
            )

        # Test service-level validation for zero quantity
        # This raises our custom ValidationError (not Pydantic's) during OrderRequest construction
        with pytest.raises(ValidationError):
            zero_quantity_order = OrderRequest(
                symbol="BTCUSDT",
                side="buy",
                order_type="limit",
                quantity=Decimal("0"),  # Zero quantity fails custom validator
                price=Decimal("50000.0"),
            )

        await service.stop()
