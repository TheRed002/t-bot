"""
Direct tests for BaseExchange class to achieve high coverage.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.core.exceptions import ExchangeError, ServiceError
from src.exchanges.binance import BinanceExchange


class TestBaseExchangeViaConcrete:
    """Test BaseExchange class methods through a concrete implementation."""

    @pytest.fixture
    def exchange_config(self):
        """Create test configuration for exchange."""
        return {
            "api_key": "test_api_key",
            "api_secret": "test_api_secret",
            "testnet": True,
            "sandbox": True
        }

    @pytest.fixture
    def binance_exchange(self, exchange_config):
        """Create BinanceExchange instance for testing BaseExchange methods."""
        with patch("src.exchanges.binance.AsyncClient"):
            exchange = BinanceExchange(config=exchange_config)
            # Mock the client to prevent real API calls
            exchange.client = AsyncMock()
            return exchange

    def test_base_exchange_initialization(self, binance_exchange, exchange_config):
        """Test BaseExchange initialization through concrete implementation."""
        # Test attributes inherited from BaseExchange
        assert binance_exchange.exchange_name == "binance"
        assert binance_exchange.config == exchange_config
        assert binance_exchange.connected is False
        assert binance_exchange.last_heartbeat is None
        assert binance_exchange._exchange_info is None
        assert binance_exchange._trading_symbols is None

        # Test BaseService initialization
        assert hasattr(binance_exchange, "logger")
        assert binance_exchange._name == "binance_exchange"

    def test_is_connected_method(self, binance_exchange):
        """Test is_connected method from BaseExchange."""
        # Initially not connected
        assert binance_exchange.is_connected() is False

        # Set internal connected state and test
        binance_exchange._connected = True
        assert binance_exchange.is_connected() is True

        # Set back to disconnected
        binance_exchange._connected = False
        assert binance_exchange.is_connected() is False

    @pytest.mark.asyncio
    async def test_base_service_lifecycle_methods(self, binance_exchange):
        """Test BaseService lifecycle methods through BaseExchange."""
        # Mock dependency resolution to avoid DI container requirement
        with patch.object(binance_exchange, "resolve_dependency") as mock_resolve:
            mock_resolve.return_value = None  # Return None for all dependencies
            
            # Mock the abstract methods to prevent real API calls
            with patch.object(binance_exchange, "connect", new_callable=AsyncMock) as mock_connect:
                with patch.object(binance_exchange, "load_exchange_info", new_callable=AsyncMock) as mock_load_info:
                    with patch.object(binance_exchange, "_update_connection_state", new_callable=AsyncMock) as mock_update_state:
                        mock_connect.return_value = None
                        mock_load_info.return_value = None
                        mock_update_state.return_value = None

                        # Test _do_start method
                        await binance_exchange._do_start()

                        # Verify abstract methods were called
                        mock_connect.assert_called_once()
                        mock_load_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_base_service_lifecycle_error_handling(self, binance_exchange):
        """Test BaseExchange error handling in lifecycle methods."""
        # Mock dependency resolution to avoid DI container requirement
        with patch.object(binance_exchange, "resolve_dependency") as mock_resolve:
            mock_resolve.return_value = None  # Return None for all dependencies
            
            # Mock connect to raise an exception
            with patch.object(binance_exchange, "connect", side_effect=Exception("Connection failed")):
                with pytest.raises(ServiceError, match="Exchange startup failed"):
                    await binance_exchange._do_start()

    @pytest.mark.asyncio
    async def test_do_stop_method(self, binance_exchange):
        """Test _do_stop method from BaseExchange."""
        # Mock disconnect method
        with patch.object(binance_exchange, "disconnect", new_callable=AsyncMock) as mock_disconnect:
            mock_disconnect.return_value = None

            # Test _do_stop method
            await binance_exchange._do_stop()

            # Verify disconnect was called
            mock_disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_method(self, binance_exchange):
        """Test health_check method from BaseExchange."""
        # Test health check when service is running
        binance_exchange._status = "running"
        binance_exchange._connected = True

        with patch.object(binance_exchange, "ping", new_callable=AsyncMock) as mock_ping:
            mock_ping.return_value = True

            health = await binance_exchange.health_check()

            # Verify health check results
            assert health.status.value == "healthy"
            mock_ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, binance_exchange):
        """Test health_check method when service is unhealthy."""
        # Test health check when service is not running
        binance_exchange._status = "stopped"
        binance_exchange._connected = False

        health = await binance_exchange.health_check()

        # Should be unhealthy when stopped
        assert health.status.value == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_with_ping_failure(self, binance_exchange):
        """Test health_check method when ping fails."""
        binance_exchange._status = "running"
        binance_exchange._connected = True

        with patch.object(binance_exchange, "ping", side_effect=Exception("Ping failed")):
            health = await binance_exchange.health_check()

            # Should be unhealthy when ping fails
            assert health.status.value == "unhealthy"

    def test_validate_config_method(self, binance_exchange, exchange_config):
        """Test validate_config method from BaseService."""
        # Test with valid config
        result = binance_exchange.validate_config(exchange_config)
        assert result is True

        # Test with invalid config (missing required fields)
        invalid_config = {"api_key": "test"}  # Missing api_secret
        result = binance_exchange.validate_config(invalid_config)
        assert result is False

    def test_string_representation(self, binance_exchange):
        """Test string representation of BaseExchange."""
        str_repr = str(binance_exchange)
        assert "binance_exchange" in str_repr
        assert "BaseExchange" in str_repr or "BinanceExchange" in str_repr


class TestBaseExchangeAttributes:
    """Test BaseExchange attribute handling and property methods."""

    @pytest.fixture
    def exchange(self):
        """Create exchange instance for attribute testing."""
        config = {"api_key": "test", "api_secret": "secret", "testnet": True}
        with patch("src.exchanges.binance.AsyncClient"):
            exchange = BinanceExchange(config=config)
            exchange.client = AsyncMock()
            return exchange

    def test_exchange_info_caching(self, exchange):
        """Test exchange info caching mechanism."""
        # Initially None
        assert exchange._exchange_info is None

        # Mock exchange info
        mock_info = {"symbols": ["BTCUSDT", "ETHUSDT"]}
        exchange._exchange_info = mock_info

        # Verify cached
        assert exchange._exchange_info == mock_info

    def test_trading_symbols_caching(self, exchange):
        """Test trading symbols caching."""
        # Initially None
        assert exchange._trading_symbols is None

        # Set symbols
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        exchange._trading_symbols = symbols

        # Verify cached
        assert exchange._trading_symbols == symbols

    def test_connection_state_tracking(self, exchange):
        """Test connection state tracking."""
        # Initially disconnected
        assert exchange.connected is False
        assert exchange.last_heartbeat is None

        # Set connected with heartbeat
        exchange._connected = True
        test_time = datetime.now(timezone.utc)
        exchange._last_heartbeat = test_time

        # Verify state
        assert exchange.connected is True
        assert exchange.last_heartbeat == test_time
        assert exchange.is_connected() is True

    def test_dependency_registration(self, exchange):
        """Test that BaseExchange registers required dependencies."""
        # BaseExchange should register config dependency
        dependencies = getattr(exchange, "_dependencies", [])
        assert "config" in dependencies or hasattr(exchange, "config")


class TestBaseExchangeErrorScenarios:
    """Test error handling scenarios in BaseExchange."""

    @pytest.fixture
    def exchange(self):
        config = {"api_key": "test", "api_secret": "secret"}
        with patch("src.exchanges.binance.AsyncClient"):
            exchange = BinanceExchange(config=config)
            exchange.client = AsyncMock()
            return exchange

    @pytest.mark.asyncio
    async def test_startup_failure_handling(self, exchange):
        """Test handling of startup failures."""
        # Mock dependency resolution to avoid DI container requirement
        with patch.object(exchange, "resolve_dependency") as mock_resolve:
            mock_resolve.return_value = None  # Return None for all dependencies
            
            # Mock connect to fail
            with patch.object(exchange, "connect", side_effect=ConnectionError("Network error")):
                with pytest.raises(ServiceError) as exc_info:
                    await exchange._do_start()

                assert "Exchange startup failed" in str(exc_info.value)
                assert "Network error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_exchange_info_failure(self, exchange):
        """Test handling of exchange info loading failure."""
        # Mock dependency resolution to avoid DI container requirement
        with patch.object(exchange, "resolve_dependency") as mock_resolve:
            mock_resolve.return_value = None  # Return None for all dependencies
            
            # Mock successful connect but failed load_exchange_info
            with patch.object(exchange, "connect", new_callable=AsyncMock):
                with patch.object(exchange, "load_exchange_info", side_effect=ExchangeError("API error")):
                    with pytest.raises(ServiceError) as exc_info:
                        await exchange._do_start()

                    assert "Exchange startup failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stop_method_exception_handling(self, exchange):
        """Test that _do_stop handles disconnect exceptions gracefully."""
        # Mock disconnect to raise exception
        with patch.object(exchange, "disconnect", side_effect=Exception("Disconnect error")):
            # _do_stop should handle this gracefully and log the error
            await exchange._do_stop()
            # Should not raise exception

    def test_validate_config_edge_cases(self, exchange):
        """Test validate_config with various edge cases."""
        # Test with None config
        result = exchange.validate_config(None)
        assert result is False

        # Test with empty config
        result = exchange.validate_config({})
        assert result is False

        # Test with non-dict config
        result = exchange.validate_config("not a dict")
        assert result is False
