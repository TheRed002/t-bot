"""
Unit tests for Binance Exchange Implementation

These tests verify the BinanceExchange class implementation
which inherits from BaseExchange and provides production-ready
Binance API integration.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.exceptions import ServiceError, ValidationError
from src.core.types import OrderRequest, OrderSide, OrderStatus, OrderType
from src.exchanges.binance import BinanceExchange


class TestBinanceExchangeImplementation:
    """Test suite for BinanceExchange implementation."""

    @pytest.fixture
    def valid_config(self):
        """Create valid configuration."""
        return {
            "api_key": "test_api_key",
            "api_secret": "test_api_secret",
            "testnet": True
        }

    @pytest.fixture
    def binance_exchange(self, valid_config):
        """Create BinanceExchange with valid configuration."""
        with patch('src.exchanges.binance.BINANCE_AVAILABLE', True):
            return BinanceExchange(config=valid_config)

    def test_initialization_with_config(self, valid_config):
        """Test initialization with valid configuration."""
        with patch('src.exchanges.binance.BINANCE_AVAILABLE', True):
            exchange = BinanceExchange(config=valid_config)

            assert exchange.name == "binance_exchange"
            assert exchange.api_key == "test_api_key"
            assert exchange.api_secret == "test_api_secret"
            assert exchange.testnet is True
            assert exchange.client is None  # Not connected yet

    def test_initialization_missing_api_key(self):
        """Test initialization fails without API key."""
        config = {
            "api_secret": "test_api_secret",
            "testnet": True
        }
        
        with patch('src.exchanges.binance.BINANCE_AVAILABLE', True):
            with pytest.raises(ValidationError, match="Binance API key and secret are required"):
                BinanceExchange(config=config)

    def test_initialization_missing_api_secret(self):
        """Test initialization fails without API secret."""
        config = {
            "api_key": "test_api_key",
            "testnet": True
        }
        
        with patch('src.exchanges.binance.BINANCE_AVAILABLE', True):
            with pytest.raises(ValidationError, match="Binance API key and secret are required"):
                BinanceExchange(config=config)

    def test_initialization_binance_not_available(self, valid_config):
        """Test initialization fails when Binance SDK not available."""
        with patch('src.exchanges.binance.BINANCE_AVAILABLE', False):
            with pytest.raises(ServiceError, match="Binance SDK not available"):
                BinanceExchange(config=valid_config)

    def test_validate_service_config(self, binance_exchange):
        """Test configuration validation."""
        # Test valid config
        valid_config = {
            "api_key": "test_key",
            "api_secret": "test_secret"
        }
        assert binance_exchange._validate_service_config(valid_config) is True

        # Test invalid configs
        assert binance_exchange._validate_service_config({}) is False
        assert binance_exchange._validate_service_config({"api_key": "test"}) is False
        assert binance_exchange._validate_service_config({"api_secret": "test"}) is False
        assert binance_exchange._validate_service_config({"api_key": 123, "api_secret": "test"}) is False

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, binance_exchange):
        """Test connection and disconnection methods."""
        # Mock the Binance AsyncClient
        mock_client = AsyncMock()
        mock_client.get_account = AsyncMock(return_value={"accountType": "SPOT"})
        mock_client.close_connection = AsyncMock()
        mock_client.ping = AsyncMock()
        
        with patch('src.exchanges.binance.AsyncClient', return_value=mock_client), \
             patch.object(binance_exchange, 'load_exchange_info', new_callable=AsyncMock):
            # Test connection
            await binance_exchange.connect()
            assert binance_exchange.is_connected() is True
            assert binance_exchange.client is not None

            # Test ping
            result = await binance_exchange.ping()
            assert result is True

            # Test disconnection
            await binance_exchange.disconnect()
            assert binance_exchange.is_connected() is False
            assert binance_exchange.client is None

    @pytest.mark.asyncio
    async def test_load_exchange_info(self, binance_exchange):
        """Test loading exchange information."""
        mock_client = AsyncMock()
        mock_exchange_info = {
            "timezone": "UTC",
            "serverTime": 1234567890,
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "filters": [
                        {
                            "filterType": "PRICE_FILTER",
                            "minPrice": "0.01",
                            "maxPrice": "1000000",
                            "tickSize": "0.01"
                        },
                        {
                            "filterType": "LOT_SIZE",
                            "minQty": "0.00001",
                            "maxQty": "10000",
                            "stepSize": "0.00001"
                        }
                    ]
                }
            ]
        }
        mock_client.get_exchange_info = AsyncMock(return_value=mock_exchange_info)
        binance_exchange.client = mock_client

        exchange_info = await binance_exchange.load_exchange_info()

        # The returned ExchangeInfo is for BTCUSDT symbol
        assert exchange_info.symbol == "BTCUSDT"
        assert exchange_info.base_asset == "BTC"
        assert exchange_info.quote_asset == "USDT"
        assert exchange_info.exchange == "binance"
        assert exchange_info.status == "TRADING"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])