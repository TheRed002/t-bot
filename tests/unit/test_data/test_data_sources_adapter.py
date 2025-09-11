"""
Test suite for data sources adapter.

This module contains comprehensive tests for the DataSourceAdapter
including source creation, parameter adaptation, response standardization,
and symbol/timeframe conversion.
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.exceptions import ConfigurationError, DataError
from src.data.interfaces import DataSourceInterface
from src.data.sources.adapter import DataSourceAdapter


@pytest.fixture
def mock_data_source():
    """Mock data source for testing."""
    source = Mock(spec=DataSourceInterface)
    source.fetch = AsyncMock()
    source.stream = AsyncMock()
    source.connect = AsyncMock()
    source.disconnect = AsyncMock()
    source.is_connected = Mock(return_value=True)
    return source


@pytest.fixture
def sample_binance_kline_list():
    """Sample Binance kline data as list format."""
    return [
        1640995200000,  # timestamp
        "50000.00",  # open
        "50500.00",  # high
        "49500.00",  # low
        "50200.00",  # close
        "1.50000000",  # volume
        1640995259999,  # close timestamp
        "75300.00",  # quote asset volume
        100,  # number of trades
        "0.75000000",  # taker buy base volume
        "37650.00",  # taker buy quote volume
        "0",  # ignore
    ]


@pytest.fixture
def sample_binance_kline_dict():
    """Sample Binance kline data as dictionary format."""
    return {
        "openTime": 1640995200000,
        "open": "50000.00",
        "high": "50500.00",
        "low": "49500.00",
        "close": "50200.00",
        "volume": "1.50000000",
        "closeTime": 1640995259999,
        "quoteAssetVolume": "75300.00",
        "numberOfTrades": 100,
        "takerBuyBaseAssetVolume": "0.75000000",
        "takerBuyQuoteAssetVolume": "37650.00",
    }


@pytest.fixture
def sample_coinbase_candle():
    """Sample Coinbase candle data."""
    return {
        "time": 1640995200,
        "open": "50000.00",
        "high": "50500.00",
        "low": "49500.00",
        "close": "50200.00",
        "volume": "1.50000000",
    }


@pytest.fixture
def sample_okx_candle_list():
    """Sample OKX candle data as list format."""
    return [
        "1640995200000",  # timestamp
        "50000.00",  # open
        "50500.00",  # high
        "49500.00",  # low
        "50200.00",  # close
        "1.50000000",  # volume
        "75300.00",  # quote volume
    ]


class TestDataSourceAdapter:
    """Test DataSourceAdapter class."""

    def test_initialization_binance(self):
        """Test adapter initialization with Binance source."""
        mock_source = Mock(spec=DataSourceInterface)

        with patch.object(DataSourceAdapter, "_create_source", return_value=mock_source):
            adapter = DataSourceAdapter("binance", api_key="test_key")

            assert adapter.source_type == "binance"
            assert adapter.config == {"api_key": "test_key"}
            assert adapter.source == mock_source

    def test_initialization_coinbase(self):
        """Test adapter initialization with Coinbase source."""
        mock_source = Mock(spec=DataSourceInterface)

        with patch.object(DataSourceAdapter, "_create_source", return_value=mock_source):
            adapter = DataSourceAdapter("coinbase", api_key="test_key")

            assert adapter.source_type == "coinbase"
            assert adapter.config == {"api_key": "test_key"}
            assert adapter.source == mock_source

    def test_initialization_okx(self):
        """Test adapter initialization with OKX source."""
        mock_source = Mock(spec=DataSourceInterface)

        with patch.object(DataSourceAdapter, "_create_source", return_value=mock_source):
            adapter = DataSourceAdapter("OKX", api_key="test_key")

            assert adapter.source_type == "okx"  # Should be lowercased
            assert adapter.config == {"api_key": "test_key"}
            assert adapter.source == mock_source

    def test_initialization_unsupported_source(self):
        """Test adapter initialization with unsupported source type."""
        with pytest.raises(ConfigurationError, match="Unsupported data source type: unsupported"):
            DataSourceAdapter("unsupported")

    @pytest.mark.asyncio
    async def test_fetch_market_data_success(self, mock_data_source):
        """Test successful market data fetching."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=mock_data_source):
            adapter = DataSourceAdapter("binance")

            # Mock raw data response
            raw_data = [{"timestamp": 1640995200000, "open": "50000", "close": "50200"}]
            mock_data_source.fetch.return_value = raw_data

            # Mock standardization
            with patch.object(adapter, "_standardize_response") as mock_standardize:
                mock_standardize.return_value = [{"standardized": True}]

                result = await adapter.fetch_market_data("BTC/USDT", "1h", 100)

                assert result == [{"standardized": True}]
                mock_data_source.fetch.assert_called_once()
                mock_standardize.assert_called_once_with(raw_data)

    @pytest.mark.asyncio
    async def test_stream_market_data_success(self, mock_data_source):
        """Test successful market data streaming."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=mock_data_source):
            adapter = DataSourceAdapter("binance")

            # Create async iterator for streaming
            async def mock_stream():
                yield {"timestamp": 1640995200000, "price": "50000"}
                yield {"timestamp": 1640995260000, "price": "50100"}

            mock_data_source.stream.return_value = mock_stream()

            # Mock standardization
            with patch.object(adapter, "_standardize_record") as mock_standardize:
                mock_standardize.side_effect = [
                    {"standardized": True, "price": "50000"},
                    {"standardized": True, "price": "50100"},
                ]

                results = []
                async for record in adapter.stream_market_data("BTC/USDT"):
                    results.append(record)

                assert len(results) == 2
                assert all(r["standardized"] for r in results)
                assert mock_standardize.call_count == 2

    def test_adapt_fetch_params_binance(self):
        """Test parameter adaptation for Binance."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("binance")

            params = adapter._adapt_fetch_params("BTC/USDT", "1h", 100, extra_param="test")

            expected = {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "limit": 100,
                "extra_param": "test",
            }
            assert params == expected

    def test_adapt_fetch_params_coinbase(self):
        """Test parameter adaptation for Coinbase."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("coinbase")

            params = adapter._adapt_fetch_params("BTC/USD", "1h", 100)

            expected = {
                "product_id": "BTC-USD",
                "granularity": 3600,
                "limit": 100,
            }
            assert params == expected

    def test_adapt_fetch_params_okx(self):
        """Test parameter adaptation for OKX."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("okx")

            params = adapter._adapt_fetch_params("BTC/USDT", "1h", 100)

            expected = {
                "instId": "BTC-USDT",
                "bar": "1h",
                "limit": "100",
            }
            assert params == expected

    def test_adapt_fetch_params_unknown_source(self):
        """Test parameter adaptation for unknown source (pass-through)."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter.__new__(DataSourceAdapter)  # Skip init
            adapter.source_type = "unknown"

            params = adapter._adapt_fetch_params("BTC/USDT", "1h", 100)

            expected = {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "limit": 100,
            }
            assert params == expected

    def test_adapt_stream_params_binance(self):
        """Test stream parameter adaptation for Binance."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("binance")

            params = adapter._adapt_stream_params("BTC/USDT", extra="test")

            expected = {"symbol": "BTCUSDT", "extra": "test"}
            assert params == expected

    def test_adapt_stream_params_coinbase(self):
        """Test stream parameter adaptation for Coinbase."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("coinbase")

            params = adapter._adapt_stream_params("BTC/USD")

            expected = {"product_ids": ["BTC-USD"]}
            assert params == expected

    def test_adapt_stream_params_okx(self):
        """Test stream parameter adaptation for OKX."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("okx")

            params = adapter._adapt_stream_params("BTC/USDT")

            expected = {"instId": "BTC-USDT"}
            assert params == expected

    def test_standardize_response(self):
        """Test response standardization."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("binance")

            raw_data = [{"raw": "data1"}, {"raw": "data2"}]

            with patch.object(adapter, "_standardize_record") as mock_standardize:
                mock_standardize.side_effect = [{"std": "data1"}, {"std": "data2"}]

                result = adapter._standardize_response(raw_data)

                expected = [{"std": "data1"}, {"std": "data2"}]
                assert result == expected
                assert mock_standardize.call_count == 2

    def test_standardize_record_binance_list(self, sample_binance_kline_list):
        """Test record standardization for Binance list format."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("binance")

            result = adapter._standardize_record(sample_binance_kline_list)

            assert result["timestamp"] == 1640995200000
            assert result["open"] == Decimal("50000.00")
            assert result["high"] == Decimal("50500.00")
            assert result["low"] == Decimal("49500.00")
            assert result["close"] == Decimal("50200.00")
            assert result["volume"] == Decimal("1.50000000")
            assert result["source"] == "binance"

    def test_standardize_record_binance_dict(self, sample_binance_kline_dict):
        """Test record standardization for Binance dictionary format."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("binance")

            result = adapter._standardize_record(sample_binance_kline_dict)

            assert result["timestamp"] == 1640995200000
            assert result["open"] == Decimal("50000.00")
            assert result["high"] == Decimal("50500.00")
            assert result["low"] == Decimal("49500.00")
            assert result["close"] == Decimal("50200.00")
            assert result["volume"] == Decimal("1.50000000")
            assert result["source"] == "binance"

    def test_standardize_record_coinbase(self, sample_coinbase_candle):
        """Test record standardization for Coinbase format."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("coinbase")

            result = adapter._standardize_record(sample_coinbase_candle)

            assert result["timestamp"] == 1640995200
            assert result["open"] == Decimal("50000.00")
            assert result["high"] == Decimal("50500.00")
            assert result["low"] == Decimal("49500.00")
            assert result["close"] == Decimal("50200.00")
            assert result["volume"] == Decimal("1.50000000")
            assert result["source"] == "coinbase"

    def test_standardize_record_okx_list(self, sample_okx_candle_list):
        """Test record standardization for OKX list format."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("okx")

            result = adapter._standardize_record(sample_okx_candle_list)

            assert result["timestamp"] == 1640995200000
            assert result["open"] == Decimal("50000.00")
            assert result["high"] == Decimal("50500.00")
            assert result["low"] == Decimal("49500.00")
            assert result["close"] == Decimal("50200.00")
            assert result["volume"] == Decimal("1.50000000")
            assert result["source"] == "okx"

    def test_standardize_record_okx_dict(self):
        """Test record standardization for OKX dictionary format."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("okx")

            sample_dict = {"existing": "data"}
            result = adapter._standardize_record(sample_dict)

            assert result["existing"] == "data"
            assert result["source"] == "okx"

    def test_standardize_record_unknown_source(self):
        """Test record standardization for unknown source."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter.__new__(DataSourceAdapter)  # Skip init
            adapter.source_type = "unknown"

            sample_record = {"existing": "data"}
            result = adapter._standardize_record(sample_record)

            assert result["existing"] == "data"
            assert result["source"] == "unknown"

    def test_symbol_to_binance(self):
        """Test symbol conversion to Binance format."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("binance")

            assert adapter._symbol_to_binance("BTC/USDT") == "BTCUSDT"
            assert adapter._symbol_to_binance("btc-usdt") == "BTCUSDT"
            assert adapter._symbol_to_binance("BTC_USDT") == "BTCUSDT"
            assert adapter._symbol_to_binance("BTCUSDT") == "BTCUSDT"

    def test_symbol_to_coinbase_pair(self):
        """Test symbol conversion to Coinbase format."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("coinbase")

            assert adapter._symbol_to_coinbase_pair("BTC/USD") == "BTC-USD"
            assert adapter._symbol_to_coinbase_pair("BTC_USD") == "BTC-USD"
            assert adapter._symbol_to_coinbase_pair("btc/usd") == "BTC-USD"
            assert adapter._symbol_to_coinbase_pair("BTC-USD") == "BTC-USD"

    def test_symbol_to_okx_inst(self):
        """Test symbol conversion to OKX format."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("okx")

            assert adapter._symbol_to_okx_inst("BTC/USDT") == "BTC-USDT"
            assert adapter._symbol_to_okx_inst("BTC_USDT") == "BTC-USDT"

    def test_timeframe_to_binance_interval(self):
        """Test timeframe conversion to Binance interval."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("binance")

            assert adapter._timeframe_to_binance_interval("1m") == "1m"
            assert adapter._timeframe_to_binance_interval("1h") == "1h"
            assert adapter._timeframe_to_binance_interval("1d") == "1d"
            assert adapter._timeframe_to_binance_interval("1w") == "1w"
            assert adapter._timeframe_to_binance_interval("unknown") == "1h"  # default

    def test_timeframe_to_coinbase_granularity(self):
        """Test timeframe conversion to Coinbase granularity."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("coinbase")

            assert adapter._timeframe_to_coinbase_granularity("1m") == 60
            assert adapter._timeframe_to_coinbase_granularity("5m") == 300
            assert adapter._timeframe_to_coinbase_granularity("1h") == 3600
            assert adapter._timeframe_to_coinbase_granularity("1d") == 86400
            assert adapter._timeframe_to_coinbase_granularity("unknown") == 3600  # default

    def test_timeframe_to_okx_bar(self):
        """Test timeframe conversion to OKX bar format."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("okx")

            assert adapter._timeframe_to_okx_bar("1m") == "1m"
            assert adapter._timeframe_to_okx_bar("1h") == "1h"
            assert adapter._timeframe_to_okx_bar("1d") == "1d"

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_data_source):
        """Test successful connection."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=mock_data_source):
            adapter = DataSourceAdapter("binance")

            await adapter.connect()

            mock_data_source.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_timeout(self, mock_data_source):
        """Test connection timeout."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=mock_data_source):
            adapter = DataSourceAdapter("binance")

            # Mock connection to hang indefinitely
            async def hang_forever():
                await asyncio.sleep(60)

            mock_data_source.connect.side_effect = hang_forever

            with pytest.raises(DataError, match="Connection timeout for binance data source"):
                await adapter.connect()

    @pytest.mark.asyncio
    async def test_disconnect_success(self, mock_data_source):
        """Test successful disconnection."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=mock_data_source):
            adapter = DataSourceAdapter("binance")

            await adapter.disconnect()

            mock_data_source.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_timeout(self, mock_data_source):
        """Test disconnect timeout."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=mock_data_source):
            adapter = DataSourceAdapter("binance")

            # Mock disconnect to hang indefinitely
            mock_data_source.disconnect.side_effect = asyncio.sleep(60)

            # Should not raise exception, just log warning
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_error(self, mock_data_source):
        """Test disconnect error handling."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=mock_data_source):
            adapter = DataSourceAdapter("binance")

            mock_data_source.disconnect.side_effect = Exception("Disconnect error")

            # Should not raise exception, just log warning
            await adapter.disconnect()

    def test_is_connected(self, mock_data_source):
        """Test connection status check."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=mock_data_source):
            adapter = DataSourceAdapter("binance")

            mock_data_source.is_connected.return_value = True
            assert adapter.is_connected() is True

            mock_data_source.is_connected.return_value = False
            assert adapter.is_connected() is False


class TestSymbolConversionEdgeCases:
    """Test edge cases in symbol conversion methods."""

    def test_symbol_conversion_edge_cases(self):
        """Test various edge cases in symbol conversion."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("binance")

            # Test empty/None cases (should not crash)
            assert adapter._symbol_to_binance("") == ""

            # Test complex symbols
            assert adapter._symbol_to_binance("BTC/USDT-PERP") == "BTCUSDTPERP"

            # Test already formatted symbols
            assert adapter._symbol_to_coinbase_pair("BTC-USD") == "BTC-USD"

            # Test single part symbols
            parts_result = adapter._symbol_to_coinbase_pair("BTCUSD")
            assert parts_result == "BTCUSD"  # No split possible


class TestDecimalPrecisionHandling:
    """Test decimal precision handling in standardization."""

    def test_decimal_precision_binance(self):
        """Test decimal precision is maintained for Binance data."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("binance")

            # Test with high precision values
            sample_data = [
                1640995200000,
                "50000.12345678",  # 8 decimal places
                "50500.87654321",
                "49500.11111111",
                "50200.99999999",
                "1.23456789",
            ]

            result = adapter._standardize_record(sample_data)

            # Check that precision is preserved with quantization
            assert str(result["open"]) == "50000.12345678"
            assert str(result["high"]) == "50500.87654321"
            assert str(result["volume"]) == "1.23456789"

    def test_decimal_precision_with_missing_fields(self):
        """Test decimal handling with missing fields."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=Mock()):
            adapter = DataSourceAdapter("binance")

            # Test with dictionary missing some fields
            sample_data = {
                "openTime": 1640995200000,
                "open": "50000.00",
                # missing high, low, close, volume
            }

            result = adapter._standardize_record(sample_data)

            assert result["open"] == Decimal("50000.00")
            assert result["high"] == Decimal("0")  # default
            assert result["low"] == Decimal("0")  # default
            assert result["close"] == Decimal("0")  # default
            assert result["volume"] == Decimal("0")  # default


class TestStreamingEdgeCases:
    """Test edge cases in streaming functionality."""

    @pytest.mark.asyncio
    async def test_stream_empty_iterator(self, mock_data_source):
        """Test streaming with empty iterator."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=mock_data_source):
            adapter = DataSourceAdapter("binance")

            # Create empty async iterator
            async def empty_stream():
                return
                yield  # unreachable

            mock_data_source.stream.return_value = empty_stream()

            results = []
            async for record in adapter.stream_market_data("BTC/USDT"):
                results.append(record)

            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_stream_with_errors(self, mock_data_source):
        """Test streaming with error handling."""
        with patch.object(DataSourceAdapter, "_create_source", return_value=mock_data_source):
            adapter = DataSourceAdapter("binance")

            # Create stream that raises error
            async def error_stream():
                yield {"valid": "data"}
                raise Exception("Stream error")

            mock_data_source.stream.return_value = error_stream()

            with patch.object(adapter, "_standardize_record", return_value={"std": "data"}):
                results = []
                with pytest.raises(Exception, match="Stream error"):
                    async for record in adapter.stream_market_data("BTC/USDT"):
                        results.append(record)

                assert len(results) == 1  # Got one record before error
