"""Test suite for market data source components."""

import asyncio
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

from src.core.config import Config
from src.core.exceptions import DataSourceError
from src.core.types import Ticker
from src.data.sources.market_data import (
    DataStreamType,
    DataSubscription,
    MarketDataSource,
)


class TestDataSubscription:
    """Test suite for DataSubscription."""

    def test_initialization_minimal(self):
        """Test minimal initialization."""
        callback = Mock()
        
        subscription = DataSubscription(
            exchange_name="binance",
            symbol="BTCUSDT",
            stream_type=DataStreamType.TICKER,
            callback=callback
        )
        
        assert subscription.exchange_name == "binance"
        assert subscription.symbol == "BTCUSDT"
        assert subscription.stream_type == DataStreamType.TICKER
        assert subscription.callback is callback
        assert subscription.active is True
        assert subscription.last_update is None
        assert subscription.error_count == 0

    def test_initialization_full(self):
        """Test full initialization."""
        callback = Mock()
        timestamp = datetime.now(timezone.utc)
        
        subscription = DataSubscription(
            exchange_name="okx",
            symbol="ETHUSDT",
            stream_type=DataStreamType.ORDER_BOOK,
            callback=callback,
            active=False,
            last_update=timestamp,
            error_count=5
        )
        
        assert subscription.exchange_name == "okx"
        assert subscription.symbol == "ETHUSDT"
        assert subscription.stream_type == DataStreamType.ORDER_BOOK
        assert subscription.callback is callback
        assert subscription.active is False
        assert subscription.last_update == timestamp
        assert subscription.error_count == 5


class TestMarketDataSource:
    """Test suite for MarketDataSource."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        return config

    @pytest.fixture
    def mock_exchange_factory(self):
        """Create mock exchange factory."""
        factory = Mock()
        factory.create_exchange = AsyncMock()
        return factory

    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange."""
        exchange = Mock()
        exchange.connect = AsyncMock(return_value=True)
        exchange.get_ticker = AsyncMock()
        return exchange

    @pytest.fixture
    def market_data_source(self, mock_config, mock_exchange_factory):
        """Create market data source instance."""
        with patch('src.data.sources.market_data.ErrorHandler') as mock_error_handler_class:
            # Configure the ErrorHandler instance to have async methods
            mock_error_handler_instance = Mock()
            mock_error_handler_instance.handle_error = AsyncMock()
            mock_error_handler_instance.create_error_context = Mock()
            mock_error_handler_class.return_value = mock_error_handler_instance
            
            with patch('src.data.sources.market_data.NetworkDisconnectionRecovery'):
                with patch('src.data.sources.market_data.APIRateLimitRecovery'):
                    with patch('src.data.sources.market_data.ConnectionManager'):
                        with patch('src.data.sources.market_data.ErrorPatternAnalytics'):
                            return MarketDataSource(
                                config=mock_config,
                                exchange_factory=mock_exchange_factory
                            )

    def test_initialization(self, mock_config, mock_exchange_factory):
        """Test market data source initialization."""
        with patch('src.data.sources.market_data.ErrorHandler'):
            with patch('src.data.sources.market_data.NetworkDisconnectionRecovery'):
                with patch('src.data.sources.market_data.APIRateLimitRecovery'):
                    with patch('src.data.sources.market_data.ConnectionManager'):
                        with patch('src.data.sources.market_data.ErrorPatternAnalytics'):
                            source = MarketDataSource(
                                config=mock_config,
                                exchange_factory=mock_exchange_factory
                            )
        
        assert source.config is mock_config
        assert source.exchange_factory is mock_exchange_factory
        assert isinstance(source.exchanges, dict)
        assert isinstance(source.subscriptions, dict)
        assert isinstance(source.ticker_cache, dict)
        assert isinstance(source.order_book_cache, dict)
        assert isinstance(source.trade_cache, dict)
        assert isinstance(source.active_streams, dict)
        assert isinstance(source.stream_tasks, dict)
        assert isinstance(source.stats, dict)

    def test_initialization_without_factory(self, mock_config):
        """Test initialization without exchange factory."""
        with patch('src.data.sources.market_data.ErrorHandler'):
            with patch('src.data.sources.market_data.NetworkDisconnectionRecovery'):
                with patch('src.data.sources.market_data.APIRateLimitRecovery'):
                    with patch('src.data.sources.market_data.ConnectionManager'):
                        with patch('src.data.sources.market_data.ErrorPatternAnalytics'):
                            with patch('src.data.sources.market_data.ExchangeFactory') as mock_factory_class:
                                mock_factory_instance = Mock()
                                mock_factory_class.return_value = mock_factory_instance
                                
                                source = MarketDataSource(config=mock_config)
        
        assert source.exchange_factory is mock_factory_instance

    @pytest.mark.asyncio
    async def test_initialize_success(self, market_data_source, mock_exchange):
        """Test successful initialization."""
        market_data_source.exchange_factory.create_exchange.return_value = mock_exchange
        
        await market_data_source.initialize()
        
        assert len(market_data_source.exchanges) > 0
        # Should have attempted to create exchanges for supported exchanges
        assert market_data_source.exchange_factory.create_exchange.call_count == 3  # binance, okx, coinbase

    @pytest.mark.asyncio
    async def test_initialize_connection_timeout(self, market_data_source, mock_exchange):
        """Test initialization with connection timeout."""
        mock_exchange.connect.side_effect = asyncio.TimeoutError("Connection timeout")
        market_data_source.exchange_factory.create_exchange.return_value = mock_exchange
        
        # Should raise DataSourceError when no exchanges are connected
        with pytest.raises(DataSourceError, match="No exchanges connected for market data"):
            await market_data_source.initialize()
        
        assert len(market_data_source.exchanges) == 0

    @pytest.mark.asyncio
    async def test_initialize_exchange_connection_failure(self, market_data_source, mock_exchange):
        """Test initialization with exchange connection failure."""
        mock_exchange.connect.return_value = False  # Connection failed
        market_data_source.exchange_factory.create_exchange.return_value = mock_exchange
        
        # Should raise DataSourceError when no exchanges are connected
        with pytest.raises(DataSourceError, match="No exchanges connected for market data"):
            await market_data_source.initialize()
        
        assert len(market_data_source.exchanges) == 0

    @pytest.mark.asyncio
    async def test_initialize_no_exchanges_connected(self, market_data_source, mock_exchange):
        """Test initialization when no exchanges can connect."""
        # Mock exchange that fails to connect
        mock_exchange.connect.return_value = False
        market_data_source.exchange_factory.create_exchange.return_value = mock_exchange
        
        with pytest.raises(DataSourceError, match="No exchanges connected for market data"):
            await market_data_source.initialize()

    @pytest.mark.asyncio
    async def test_subscribe_to_ticker_success(self, market_data_source, mock_exchange):
        """Test successful ticker subscription."""
        market_data_source.exchanges["binance"] = mock_exchange
        callback = Mock()
        
        # Mock the active_streams to prevent _ticker_stream from being called
        market_data_source.active_streams["binance_ticker"] = True
        
        with patch.object(market_data_source, '_ticker_stream') as mock_stream:
            with patch('asyncio.create_task') as mock_create_task:
                subscription_id = await market_data_source.subscribe_to_ticker(
                    "binance", "BTCUSDT", callback
                )
        
        assert subscription_id == "binance_BTCUSDT_ticker"
        assert subscription_id in market_data_source.subscriptions
        
        subscription = market_data_source.subscriptions[subscription_id]
        assert subscription.exchange_name == "binance"
        assert subscription.symbol == "BTCUSDT"
        assert subscription.stream_type == DataStreamType.TICKER
        assert subscription.callback is callback

    @pytest.mark.asyncio
    async def test_subscribe_to_ticker_exchange_not_available(self, market_data_source):
        """Test ticker subscription with unavailable exchange."""
        callback = Mock()
        
        with pytest.raises(DataSourceError, match="Exchange nonexistent not available"):
            await market_data_source.subscribe_to_ticker("nonexistent", "BTCUSDT", callback)

    @pytest.mark.asyncio
    async def test_subscribe_to_ticker_existing_stream(self, market_data_source, mock_exchange):
        """Test ticker subscription with existing stream."""
        market_data_source.exchanges["binance"] = mock_exchange
        market_data_source.active_streams["binance_ticker"] = True  # Stream already active
        callback = Mock()
        
        subscription_id = await market_data_source.subscribe_to_ticker(
            "binance", "BTCUSDT", callback
        )
        
        assert subscription_id == "binance_BTCUSDT_ticker"
        # Should not create new task since stream is already active

    @pytest.mark.asyncio
    async def test_get_historical_data_exchange_not_available(self, market_data_source):
        """Test historical data retrieval with unavailable exchange."""
        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc)
        
        with pytest.raises(DataSourceError):
            await market_data_source.get_historical_data("test", "BTCUSDT", start_time, end_time)

    def test_stats_initialization(self, market_data_source):
        """Test statistics initialization."""
        stats = market_data_source.stats
        
        assert stats["successful_updates"] == 0
        assert stats["failed_updates"] == 0
        assert stats["total_subscriptions"] == 0
        assert stats["active_subscriptions"] == 0

    def test_cache_initialization(self, market_data_source):
        """Test cache initialization."""
        assert len(market_data_source.ticker_cache) == 0
        assert len(market_data_source.order_book_cache) == 0
        assert len(market_data_source.trade_cache) == 0

    def test_stream_management_initialization(self, market_data_source):
        """Test stream management initialization."""
        assert len(market_data_source.active_streams) == 0
        assert len(market_data_source.stream_tasks) == 0

    @pytest.mark.asyncio
    async def test_ticker_stream_would_be_created(self, market_data_source, mock_exchange):
        """Test that ticker stream task would be created properly."""
        market_data_source.exchanges["binance"] = mock_exchange
        callback = Mock()
        
        with patch.object(market_data_source, '_ticker_stream') as mock_stream:
            with patch('asyncio.create_task') as mock_create_task:
                mock_task = Mock()
                mock_create_task.return_value = mock_task
                
                await market_data_source.subscribe_to_ticker("binance", "BTCUSDT", callback)
                
                # Verify stream task would be created
                assert "binance_ticker" in market_data_source.active_streams
                assert market_data_source.active_streams["binance_ticker"] is True


class TestEnums:
    """Test suite for data stream enums."""

    def test_data_stream_type_values(self):
        """Test data stream type enum values."""
        assert DataStreamType.TICKER.value == "ticker"
        assert DataStreamType.ORDER_BOOK.value == "order_book"
        assert DataStreamType.TRADES.value == "trades"
        assert DataStreamType.OHLCV.value == "ohlcv"
        assert DataStreamType.MARKET_STATUS.value == "market_status"