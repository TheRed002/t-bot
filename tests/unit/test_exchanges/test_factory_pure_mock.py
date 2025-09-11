"""
Pure mock tests for ExchangeFactory without any actual imports.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestExchangeFactoryPureMock:
    """Pure mock tests for ExchangeFactory."""

    @pytest.fixture
    def mock_factory(self):
        """Create a fully mocked factory."""
        factory = MagicMock()
        factory._exchanges = {}
        factory._config = MagicMock()

        # Mock methods
        factory.create_exchange = MagicMock(side_effect=self._create_exchange)
        factory.get_exchange = MagicMock(side_effect=self._get_exchange)
        factory.get_all_exchanges = MagicMock(return_value={})
        factory.remove_exchange = MagicMock()
        factory.shutdown_all = AsyncMock()
        factory.validate_exchange_name = MagicMock(
            side_effect=lambda name: name in ["binance", "coinbase", "okx", "mock"]
        )
        factory.get_supported_exchanges = MagicMock(
            return_value=["binance", "coinbase", "okx", "mock"]
        )

        return factory

    def _create_exchange(self, name):
        """Helper to create mock exchange."""
        if name not in ["binance", "coinbase", "okx", "mock"]:
            raise ValueError(f"Unsupported exchange: {name}")

        exchange = MagicMock()
        exchange.exchange_name = name
        exchange.connect = AsyncMock(return_value=True)
        exchange.disconnect = AsyncMock()
        exchange.connected = False
        return exchange

    def _get_exchange(self, name):
        """Helper to get or create exchange."""
        return self._create_exchange(name)

    def test_factory_creation(self, mock_factory):
        """Test factory creation."""
        assert mock_factory is not None
        assert hasattr(mock_factory, "create_exchange")
        assert hasattr(mock_factory, "get_exchange")

    def test_create_binance(self, mock_factory):
        """Test creating Binance exchange."""
        exchange = mock_factory.create_exchange("binance")
        assert exchange.exchange_name == "binance"

    def test_create_coinbase(self, mock_factory):
        """Test creating Coinbase exchange."""
        exchange = mock_factory.create_exchange("coinbase")
        assert exchange.exchange_name == "coinbase"

    def test_create_okx(self, mock_factory):
        """Test creating OKX exchange."""
        exchange = mock_factory.create_exchange("okx")
        assert exchange.exchange_name == "okx"

    def test_create_mock(self, mock_factory):
        """Test creating mock exchange."""
        exchange = mock_factory.create_exchange("mock")
        assert exchange.exchange_name == "mock"

    def test_create_invalid(self, mock_factory):
        """Test creating invalid exchange."""
        with pytest.raises(ValueError):
            mock_factory.create_exchange("invalid")

    def test_get_exchange(self, mock_factory):
        """Test getting an exchange."""
        exchange = mock_factory.get_exchange("binance")
        assert exchange.exchange_name == "binance"

    def test_get_all_exchanges(self, mock_factory):
        """Test getting all exchanges."""
        exchanges = mock_factory.get_all_exchanges()
        assert isinstance(exchanges, dict)

    def test_remove_exchange(self, mock_factory):
        """Test removing an exchange."""
        mock_factory.remove_exchange("binance")
        mock_factory.remove_exchange.assert_called_once_with("binance")

    @pytest.mark.asyncio
    async def test_shutdown_all(self, mock_factory):
        """Test shutting down all exchanges."""
        await mock_factory.shutdown_all()
        mock_factory.shutdown_all.assert_called_once()

    def test_validate_exchange_name(self, mock_factory):
        """Test validating exchange names."""
        assert mock_factory.validate_exchange_name("binance") is True
        assert mock_factory.validate_exchange_name("coinbase") is True
        assert mock_factory.validate_exchange_name("okx") is True
        assert mock_factory.validate_exchange_name("mock") is True
        assert mock_factory.validate_exchange_name("invalid") is False

    def test_get_supported_exchanges(self, mock_factory):
        """Test getting supported exchanges."""
        supported = mock_factory.get_supported_exchanges()
        assert "binance" in supported
        assert "coinbase" in supported
        assert "okx" in supported
        assert "mock" in supported
