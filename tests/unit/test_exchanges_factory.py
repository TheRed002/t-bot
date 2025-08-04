"""
Unit tests for the exchange factory.

This module tests the ExchangeFactory class and related components
to ensure proper functionality and error handling.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import the components to test
from src.exchanges.factory import ExchangeFactory
from src.exchanges.base import BaseExchange
from src.core.types import ExchangeInfo, ExchangeStatus
from src.core.exceptions import ExchangeError, ValidationError
from src.core.config import Config


class MockExchange(BaseExchange):
    """Mock exchange implementation for testing."""
    
    def __init__(self, config: Config, exchange_name: str):
        super().__init__(config, exchange_name)
        self.mock_data = {
            "balances": {"BTC": Decimal("1.0"), "USDT": Decimal("10000.0")},
            "orders": {},
            "market_data": {},
            "connected": False
        }
    
    async def connect(self) -> bool:
        """Mock connection implementation."""
        self.mock_data["connected"] = True
        self.connected = True
        self.status = "connected"
        return True
    
    async def disconnect(self) -> None:
        """Mock disconnection implementation."""
        self.mock_data["connected"] = False
        self.connected = False
        self.status = "disconnected"
    
    async def get_account_balance(self) -> dict:
        """Mock balance retrieval."""
        if not self.connected:
            raise Exception("Not connected")
        return self.mock_data["balances"]
    
    async def place_order(self, order):
        """Mock order placement."""
        if not self.connected:
            raise Exception("Not connected")
        return Mock()
    
    async def cancel_order(self, order_id: str) -> bool:
        """Mock order cancellation."""
        if not self.connected:
            raise Exception("Not connected")
        return True
    
    async def get_order_status(self, order_id: str):
        """Mock order status retrieval."""
        if not self.connected:
            raise Exception("Not connected")
        return Mock()
    
    async def get_market_data(self, symbol: str, timeframe: str = "1m"):
        """Mock market data retrieval."""
        if not self.connected:
            raise Exception("Not connected")
        return Mock()
    
    async def subscribe_to_stream(self, symbol: str, callback) -> None:
        """Mock stream subscription."""
        if not self.connected:
            raise Exception("Not connected")
        pass
    
    async def get_order_book(self, symbol: str, depth: int = 10):
        """Mock order book retrieval."""
        if not self.connected:
            raise Exception("Not connected")
        return Mock()
    
    async def get_trade_history(self, symbol: str, limit: int = 100):
        """Mock trade history retrieval."""
        if not self.connected:
            raise Exception("Not connected")
        return []
    
    async def get_exchange_info(self):
        """Mock exchange info retrieval."""
        if not self.connected:
            raise Exception("Not connected")
        return Mock()
    
    async def get_ticker(self, symbol: str):
        """Mock ticker retrieval."""
        if not self.connected:
            raise Exception("Not connected")
        return Mock()


class TestExchangeFactory:
    """Test cases for the ExchangeFactory class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()
    
    @pytest.fixture
    def factory(self, config):
        """Create an exchange factory instance."""
        return ExchangeFactory(config)
    
    def test_factory_initialization(self, config):
        """Test factory initialization."""
        factory = ExchangeFactory(config)
        
        assert factory.config == config
        assert factory.error_handler is not None
        assert factory._exchange_registry == {}
        assert factory._active_exchanges == {}
        assert factory._connection_pools == {}
    
    def test_register_exchange(self, factory):
        """Test exchange registration."""
        # Register a valid exchange
        factory.register_exchange("test_exchange", MockExchange)
        assert "test_exchange" in factory._exchange_registry
        assert factory._exchange_registry["test_exchange"] == MockExchange
        
        # Try to register an invalid exchange class
        class InvalidExchange:
            pass
        
        with pytest.raises(ValidationError):
            factory.register_exchange("invalid_exchange", InvalidExchange)
    
    def test_get_supported_exchanges(self, factory):
        """Test getting supported exchanges."""
        # Initially empty
        assert factory.get_supported_exchanges() == []
        
        # Register some exchanges
        factory.register_exchange("binance", MockExchange)
        factory.register_exchange("okx", MockExchange)
        
        supported = factory.get_supported_exchanges()
        assert "binance" in supported
        assert "okx" in supported
        assert len(supported) == 2
    
    def test_is_exchange_supported(self, factory):
        """Test checking if exchange is supported."""
        assert not factory.is_exchange_supported("nonexistent")
        
        factory.register_exchange("test_exchange", MockExchange)
        assert factory.is_exchange_supported("test_exchange")
    
    @pytest.mark.asyncio
    async def test_create_exchange_success(self, factory):
        """Test successful exchange creation."""
        factory.register_exchange("test_exchange", MockExchange)
        
        exchange = await factory.create_exchange("test_exchange")
        
        assert isinstance(exchange, MockExchange)
        assert exchange.exchange_name == "test_exchange"
        assert exchange.connected
    
    @pytest.mark.asyncio
    async def test_create_exchange_not_supported(self, factory):
        """Test creating unsupported exchange."""
        with pytest.raises(ValidationError, match="not supported"):
            await factory.create_exchange("nonexistent")
    
    @pytest.mark.asyncio
    async def test_create_exchange_connection_failure(self, factory):
        """Test exchange creation with connection failure."""
        class FailingExchange(MockExchange):
            async def connect(self) -> bool:
                return False
        
        factory.register_exchange("failing_exchange", FailingExchange)
        
        with pytest.raises(ExchangeError, match="Failed to connect"):
            await factory.create_exchange("failing_exchange")
    
    @pytest.mark.asyncio
    async def test_get_exchange_existing(self, factory):
        """Test getting existing exchange."""
        factory.register_exchange("test_exchange", MockExchange)
        
        # Create exchange
        exchange1 = await factory.create_exchange("test_exchange")
        factory._active_exchanges["test_exchange"] = exchange1
        
        # Get existing exchange
        exchange2 = await factory.get_exchange("test_exchange")
        
        assert exchange2 == exchange1
    
    @pytest.mark.asyncio
    async def test_get_exchange_create_if_missing(self, factory):
        """Test getting exchange with creation."""
        factory.register_exchange("test_exchange", MockExchange)
        
        # Get exchange (should create new one)
        exchange = await factory.get_exchange("test_exchange", create_if_missing=True)
        
        assert isinstance(exchange, MockExchange)
        assert "test_exchange" in factory._active_exchanges
    
    @pytest.mark.asyncio
    async def test_get_exchange_not_create_if_missing(self, factory):
        """Test getting exchange without creation."""
        # Get exchange without creating
        exchange = await factory.get_exchange("test_exchange", create_if_missing=False)
        
        assert exchange is None
        assert "test_exchange" not in factory._active_exchanges
    
    @pytest.mark.asyncio
    async def test_get_exchange_health_check_failure(self, factory):
        """Test getting exchange with failed health check."""
        factory.register_exchange("test_exchange", MockExchange)
        
        # Create and add exchange
        exchange = await factory.create_exchange("test_exchange")
        factory._active_exchanges["test_exchange"] = exchange
        
        # Mock health check to fail
        async def mock_health_check():
            return False
        
        exchange.health_check = mock_health_check
        
        # Get exchange (should create new one due to health check failure)
        new_exchange = await factory.get_exchange("test_exchange")
        
        assert new_exchange != exchange
        assert new_exchange.connected
    
    @pytest.mark.asyncio
    async def test_remove_exchange(self, factory):
        """Test removing exchange."""
        factory.register_exchange("test_exchange", MockExchange)
        
        # Create and add exchange
        exchange = await factory.create_exchange("test_exchange")
        factory._active_exchanges["test_exchange"] = exchange
        
        # Remove exchange
        result = await factory.remove_exchange("test_exchange")
        
        assert result
        assert "test_exchange" not in factory._active_exchanges
    
    @pytest.mark.asyncio
    async def test_remove_exchange_not_found(self, factory):
        """Test removing non-existent exchange."""
        result = await factory.remove_exchange("nonexistent")
        assert not result
    
    @pytest.mark.asyncio
    async def test_get_all_active_exchanges(self, factory):
        """Test getting all active exchanges."""
        factory.register_exchange("test1", MockExchange)
        factory.register_exchange("test2", MockExchange)
        
        # Create exchanges
        exchange1 = await factory.create_exchange("test1")
        exchange2 = await factory.create_exchange("test2")
        
        factory._active_exchanges["test1"] = exchange1
        factory._active_exchanges["test2"] = exchange2
        
        active_exchanges = await factory.get_all_active_exchanges()
        
        assert len(active_exchanges) == 2
        assert "test1" in active_exchanges
        assert "test2" in active_exchanges
        assert active_exchanges["test1"] == exchange1
        assert active_exchanges["test2"] == exchange2
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, factory):
        """Test health check for all exchanges."""
        factory.register_exchange("test_exchange", MockExchange)
        
        # Create exchange
        exchange = await factory.create_exchange("test_exchange")
        factory._active_exchanges["test_exchange"] = exchange
        
        # Perform health check
        health_status = await factory.health_check_all()
        
        assert "test_exchange" in health_status
        assert health_status["test_exchange"]  # Should be healthy
    
    @pytest.mark.asyncio
    async def test_disconnect_all(self, factory):
        """Test disconnecting all exchanges."""
        factory.register_exchange("test1", MockExchange)
        factory.register_exchange("test2", MockExchange)
        
        # Create exchanges
        exchange1 = await factory.create_exchange("test1")
        exchange2 = await factory.create_exchange("test2")
        
        factory._active_exchanges["test1"] = exchange1
        factory._active_exchanges["test2"] = exchange2
        
        # Disconnect all
        await factory.disconnect_all()
        
        assert len(factory._active_exchanges) == 0
    
    def test_get_exchange_info(self, factory):
        """Test getting exchange information."""
        # Test unsupported exchange
        info = factory.get_exchange_info("nonexistent")
        assert info is None
        
        # Test supported exchange
        factory.register_exchange("test_exchange", MockExchange)
        info = factory.get_exchange_info("test_exchange")
        
        assert isinstance(info, ExchangeInfo)
        assert info.name == "test_exchange"
        assert isinstance(info.rate_limits, dict)
        assert isinstance(info.features, list)
    
    def test_get_exchange_status(self, factory):
        """Test getting exchange status."""
        # Test non-existent exchange
        status = factory.get_exchange_status("nonexistent")
        assert status == ExchangeStatus.OFFLINE
        
        # Test offline exchange
        factory.register_exchange("test_exchange", MockExchange)
        status = factory.get_exchange_status("test_exchange")
        assert status == ExchangeStatus.OFFLINE
        
        # Test online exchange
        exchange = MockExchange(factory.config, "test_exchange")
        exchange.connected = True
        exchange.status = "connected"
        factory._active_exchanges["test_exchange"] = exchange
        
        status = factory.get_exchange_status("test_exchange")
        assert status == ExchangeStatus.ONLINE
    
    @pytest.mark.asyncio
    async def test_context_manager(self, factory):
        """Test async context manager functionality."""
        async with factory as f:
            assert f == factory
        
        # Should not raise any exceptions


class TestExchangeFactoryErrorHandling:
    """Test error handling in ExchangeFactory."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()
    
    @pytest.fixture
    def factory(self, config):
        """Create an exchange factory instance."""
        return ExchangeFactory(config)
    
    @pytest.mark.asyncio
    async def test_create_exchange_exception_handling(self, factory):
        """Test handling of exceptions during exchange creation."""
        class ExceptionExchange(MockExchange):
            def __init__(self, config, exchange_name):
                raise Exception("Initialization failed")
        
        factory.register_exchange("exception_exchange", ExceptionExchange)
        
        with pytest.raises(ExchangeError, match="Exchange creation failed"):
            await factory.create_exchange("exception_exchange")
    
    @pytest.mark.asyncio
    async def test_get_exchange_exception_handling(self, factory):
        """Test handling of exceptions during exchange retrieval."""
        factory.register_exchange("test_exchange", MockExchange)
        
        # Mock create_exchange to raise an exception
        async def mock_create_exchange(name):
            raise Exception("Creation failed")
        
        factory.create_exchange = mock_create_exchange
        
        # Should return None instead of raising
        result = await factory.get_exchange("test_exchange")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_remove_exchange_exception_handling(self, factory):
        """Test handling of exceptions during exchange removal."""
        factory.register_exchange("test_exchange", MockExchange)
        
        # Create exchange
        exchange = await factory.create_exchange("test_exchange")
        factory._active_exchanges["test_exchange"] = exchange
        
        # Mock disconnect to raise an exception
        async def mock_disconnect():
            raise Exception("Disconnect failed")
        
        exchange.disconnect = mock_disconnect
        
        # Should return False instead of raising
        result = await factory.remove_exchange("test_exchange")
        assert not result
    
    @pytest.mark.asyncio
    async def test_health_check_all_exception_handling(self, factory):
        """Test handling of exceptions during health checks."""
        factory.register_exchange("test_exchange", MockExchange)
        
        # Create exchange
        exchange = await factory.create_exchange("test_exchange")
        factory._active_exchanges["test_exchange"] = exchange
        
        # Mock health_check to raise an exception
        async def mock_health_check():
            raise Exception("Health check failed")
        
        exchange.health_check = mock_health_check
        
        # Should handle exception and mark as unhealthy
        health_status = await factory.health_check_all()
        assert not health_status["test_exchange"]
    
    @pytest.mark.asyncio
    async def test_disconnect_all_exception_handling(self, factory):
        """Test handling of exceptions during disconnect all."""
        factory.register_exchange("test_exchange", MockExchange)
        
        # Create exchange
        exchange = await factory.create_exchange("test_exchange")
        factory._active_exchanges["test_exchange"] = exchange
        
        # Mock disconnect to raise an exception
        async def mock_disconnect():
            raise Exception("Disconnect failed")
        
        exchange.disconnect = mock_disconnect
        
        # Should not raise exception
        await factory.disconnect_all()
        
        # Exchange should still be removed from active exchanges
        assert "test_exchange" not in factory._active_exchanges 