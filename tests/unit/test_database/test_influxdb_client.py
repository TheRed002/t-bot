"""
Unit tests for InfluxDB client.

This module tests the InfluxDBClientWrapper class and all InfluxDB-related
functionality including time series data storage, market data utilities, and querying.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from influxdb_client import Point, WritePrecision
from influxdb_client.client.exceptions import InfluxDBError

from src.core.config import Config
from src.core.exceptions import DataError, DataSourceError
from src.database.influxdb_client import InfluxDBClientWrapper


class TestInfluxDBClientWrapper:
    """Test InfluxDBClientWrapper class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=Config)
        return config

    @pytest.fixture
    def client_wrapper(self, mock_config):
        """Create InfluxDBClientWrapper instance for testing."""
        return InfluxDBClientWrapper(
            url="http://localhost:8086",
            token="test_token",
            org="test_org",
            bucket="test_bucket",
            config=mock_config
        )

    @pytest.fixture
    def client_wrapper_no_config(self):
        """Create InfluxDBClientWrapper instance without config."""
        return InfluxDBClientWrapper(
            url="http://localhost:8086",
            token="test_token",
            org="test_org",
            bucket="test_bucket"
        )

    def test_influxdb_client_init(self, mock_config):
        """Test InfluxDBClientWrapper initialization."""
        wrapper = InfluxDBClientWrapper(
            url="http://localhost:8086",
            token="test_token",
            org="test_org",
            bucket="test_bucket",
            config=mock_config
        )
        
        assert wrapper.url == "http://localhost:8086"
        assert wrapper.token == "test_token"
        assert wrapper.org == "test_org"
        assert wrapper.bucket == "test_bucket"
        assert wrapper.config == mock_config
        assert wrapper.client is None
        assert wrapper.write_api is None
        assert wrapper.query_api is None
        assert wrapper.error_handler is not None

    def test_influxdb_client_init_no_config(self):
        """Test InfluxDBClientWrapper initialization without config."""
        wrapper = InfluxDBClientWrapper(
            url="http://localhost:8086",
            token="test_token",
            org="test_org",
            bucket="test_bucket"
        )
        
        assert wrapper.config is None
        assert wrapper.error_handler is None

    @pytest.mark.asyncio
    async def test_connect_success(self, client_wrapper):
        """Test successful InfluxDB connection."""
        mock_client = Mock()
        mock_write_api = Mock()
        mock_query_api = Mock()
        mock_client.write_api.return_value = mock_write_api
        mock_client.query_api.return_value = mock_query_api
        mock_client.ping.return_value = True
        
        with patch('src.database.influxdb_client.InfluxDBClient', return_value=mock_client):
            await client_wrapper.connect()
            
            assert client_wrapper.client == mock_client
            assert client_wrapper.write_api == mock_write_api
            assert client_wrapper.query_api == mock_query_api
            mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_ping_failure(self, client_wrapper):
        """Test connection failure during ping."""
        mock_client = Mock()
        mock_client.ping.side_effect = InfluxDBError("Ping failed")
        
        with patch('src.database.influxdb_client.InfluxDBClient', return_value=mock_client):
            with pytest.raises(DataSourceError, match="InfluxDB health check failed"):
                await client_wrapper.connect()

    @pytest.mark.asyncio
    async def test_connect_client_creation_failure(self, client_wrapper):
        """Test connection failure during client creation."""
        error = InfluxDBError("Client creation failed")
        
        with patch('src.database.influxdb_client.InfluxDBClient', side_effect=error):
            with patch.object(client_wrapper.error_handler, 'handle_error', 
                            return_value=AsyncMock(return_value=False)):
                with pytest.raises(DataSourceError, match="InfluxDB connection failed"):
                    await client_wrapper.connect()

    @pytest.mark.asyncio
    async def test_connect_failure_with_recovery(self, client_wrapper):
        """Test connection failure with error handler recovery."""
        error = InfluxDBError("Connection failed")
        
        with patch('src.database.influxdb_client.InfluxDBClient', side_effect=error):
            with patch.object(client_wrapper.error_handler, 'handle_error', 
                            return_value=AsyncMock(return_value=True)) as mock_handle:
                with patch.object(client_wrapper, 'connect', 
                                side_effect=[None]) as mock_connect:
                    await client_wrapper.connect()
                    mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_no_error_handler(self, client_wrapper_no_config):
        """Test connection failure without error handler."""
        error = InfluxDBError("Connection failed")
        
        with patch('src.database.influxdb_client.InfluxDBClient', side_effect=error):
            with pytest.raises(DataSourceError, match="InfluxDB connection failed"):
                await client_wrapper_no_config.connect()

    def test_disconnect(self, client_wrapper):
        """Test InfluxDB disconnection."""
        mock_client = Mock()
        client_wrapper.client = mock_client
        
        client_wrapper.disconnect()
        
        mock_client.close.assert_called_once()

    def test_disconnect_no_client(self, client_wrapper):
        """Test disconnection when no client exists."""
        client_wrapper.client = None
        
        # Should not raise exception
        client_wrapper.disconnect()

    def test_decimal_to_float_decimal_input(self, client_wrapper):
        """Test _decimal_to_float with Decimal input."""
        decimal_val = Decimal("123.456")
        result = client_wrapper._decimal_to_float(decimal_val)
        
        assert result == 123.456
        assert isinstance(result, float)

    def test_decimal_to_float_int_input(self, client_wrapper):
        """Test _decimal_to_float with int input."""
        result = client_wrapper._decimal_to_float(42)
        
        assert result == 42.0
        assert isinstance(result, float)

    def test_decimal_to_float_float_input(self, client_wrapper):
        """Test _decimal_to_float with float input."""
        result = client_wrapper._decimal_to_float(3.14)
        
        assert result == 3.14
        assert isinstance(result, float)

    def test_decimal_to_float_none_input(self, client_wrapper):
        """Test _decimal_to_float with None input."""
        result = client_wrapper._decimal_to_float(None)
        
        assert result == 0.0
        assert isinstance(result, float)

    def test_decimal_to_float_string_input(self, client_wrapper):
        """Test _decimal_to_float with string input."""
        result = client_wrapper._decimal_to_float("123.45")
        
        assert result == 123.45
        assert isinstance(result, float)

    def test_create_point_basic(self, client_wrapper):
        """Test _create_point with basic parameters."""
        tags = {"symbol": "BTCUSDT", "exchange": "binance"}
        fields = {"price": 50000.0, "volume": 100.0}
        timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        point = client_wrapper._create_point("market_data", tags, fields, timestamp)
        
        assert isinstance(point, Point)
        # Point internal structure is complex, just verify it's created without error

    def test_create_point_no_timestamp(self, client_wrapper):
        """Test _create_point without timestamp."""
        tags = {"symbol": "BTCUSDT"}
        fields = {"price": 50000.0}
        
        with patch('src.database.influxdb_client.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_now
            mock_datetime.timezone = timezone
            
            point = client_wrapper._create_point("market_data", tags, fields)
            
            assert isinstance(point, Point)
            mock_datetime.now.assert_called_once_with(timezone.utc)

    def test_create_point_different_field_types(self, client_wrapper):
        """Test _create_point with different field types."""
        tags = {"symbol": "BTCUSDT"}
        fields = {
            "price": 50000.0,           # float
            "count": 42,                # int
            "active": True,             # bool
            "status": "open",           # str
            "metadata": {"key": "value"} # complex type -> str
        }
        
        point = client_wrapper._create_point("test_data", tags, fields)
        
        assert isinstance(point, Point)

    def test_write_point_success(self, client_wrapper):
        """Test successful point writing."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        point = Point("test_measurement").tag("symbol", "BTCUSDT").field("price", 50000.0)
        
        client_wrapper.write_point(point)
        
        mock_write_api.write.assert_called_once_with(bucket="test_bucket", record=point)

    def test_write_point_failure(self, client_wrapper):
        """Test point writing failure."""
        mock_write_api = Mock()
        mock_write_api.write.side_effect = InfluxDBError("Write failed")
        client_wrapper.write_api = mock_write_api
        
        point = Point("test_measurement").field("value", 1.0)
        
        with pytest.raises(DataError, match="Failed to write point to InfluxDB"):
            client_wrapper.write_point(point)

    def test_write_points_success(self, client_wrapper):
        """Test successful batch point writing."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        points = [
            Point("test_measurement").field("value1", 1.0),
            Point("test_measurement").field("value2", 2.0)
        ]
        
        client_wrapper.write_points(points)
        
        mock_write_api.write.assert_called_once_with(bucket="test_bucket", record=points)

    def test_write_points_failure(self, client_wrapper):
        """Test batch point writing failure."""
        mock_write_api = Mock()
        mock_write_api.write.side_effect = InfluxDBError("Batch write failed")
        client_wrapper.write_api = mock_write_api
        
        points = [Point("test").field("value", 1.0)]
        
        with pytest.raises(DataError, match="Failed to write points to InfluxDB"):
            client_wrapper.write_points(points)

    def test_write_market_data_success(self, client_wrapper):
        """Test writing market data."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        data = {
            "price": Decimal("50000.00"),
            "volume": Decimal("100.0"),
            "bid": 49950.0,
            "ask": 50050.0,
            "open": 49000.0,
            "high": 51000.0,
            "low": 48000.0
        }
        timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        client_wrapper.write_market_data("BTCUSDT", data, timestamp)
        
        mock_write_api.write.assert_called_once()
        call_args = mock_write_api.write.call_args
        assert call_args[1]["bucket"] == "test_bucket"
        assert isinstance(call_args[1]["record"], Point)

    def test_write_market_data_no_timestamp(self, client_wrapper):
        """Test writing market data without timestamp."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        data = {"price": 50000.0, "volume": 100.0}
        
        with patch.object(client_wrapper, '_create_point') as mock_create:
            mock_point = Mock()
            mock_create.return_value = mock_point
            
            client_wrapper.write_market_data("BTCUSDT", data)
            
            mock_create.assert_called_once()
            args = mock_create.call_args[0]
            assert args[0] == "market_data"  # measurement
            assert args[1]["symbol"] == "BTCUSDT"  # tags
            assert args[3] is None  # timestamp

    def test_write_market_data_batch_success(self, client_wrapper):
        """Test writing market data batch."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        data_list = [
            {
                "symbol": "BTCUSDT",
                "price": Decimal("50000.00"),
                "volume": Decimal("100.0")
            },
            {
                "symbol": "ETHUSD",
                "price": Decimal("3000.00"),
                "volume": Decimal("200.0")
            }
        ]
        
        client_wrapper.write_market_data_batch(data_list)
        
        mock_write_api.write.assert_called_once()
        call_args = mock_write_api.write.call_args
        assert call_args[1]["bucket"] == "test_bucket"
        assert isinstance(call_args[1]["record"], list)

    def test_write_market_data_batch_empty_list(self, client_wrapper):
        """Test writing empty market data batch."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        client_wrapper.write_market_data_batch([])
        
        mock_write_api.write.assert_called_once_with(bucket="test_bucket", record=[])

    def test_write_market_data_batch_invalid_data(self, client_wrapper):
        """Test writing market data batch with invalid data."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        # Non-dict data should be skipped
        data_list = [
            "invalid_data",
            {"symbol": "BTCUSDT", "price": 50000.0}
        ]
        
        client_wrapper.write_market_data_batch(data_list)
        
        mock_write_api.write.assert_called_once()
        call_args = mock_write_api.write.call_args
        points = call_args[1]["record"]
        assert len(points) == 1  # Only valid dict should create a point

    def test_decimal_precision_warning(self, client_wrapper):
        """Test that Decimal conversion is handled correctly."""
        # Test high precision Decimal
        high_precision = Decimal("50000.123456789012345")
        result = client_wrapper._decimal_to_float(high_precision)
        
        # Should be a float (precision may be lost, which is expected)
        assert isinstance(result, float)
        # Should be approximately equal (some precision loss is expected)
        assert abs(result - 50000.123456789012345) < 1e-10


class TestInfluxDBClientWrapperMarketDataUtilities:
    """Test market data specific utilities."""

    @pytest.fixture
    def client_wrapper(self):
        """Create InfluxDBClientWrapper instance for testing."""
        return InfluxDBClientWrapper(
            url="http://localhost:8086",
            token="test_token",
            org="test_org",
            bucket="test_bucket"
        )

    def test_write_market_data_missing_fields(self, client_wrapper):
        """Test writing market data with missing fields."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        # Data with only some fields
        data = {"price": 50000.0}  # Missing other fields
        
        with patch.object(client_wrapper, '_create_point') as mock_create:
            mock_point = Mock()
            mock_create.return_value = mock_point
            
            client_wrapper.write_market_data("BTCUSDT", data)
            
            # Should create point with 0 values for missing fields
            call_args = mock_create.call_args[0]
            fields = call_args[2]  # fields dict
            assert fields["price"] == 50000.0
            assert fields["volume"] == 0.0  # Should default to 0
            assert fields["bid"] == 0.0
            assert fields["ask"] == 0.0

    def test_write_market_data_all_none_values(self, client_wrapper):
        """Test writing market data with all None values."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        data = {
            "price": None,
            "volume": None,
            "bid": None,
            "ask": None,
            "open": None,
            "high": None,
            "low": None
        }
        
        with patch.object(client_wrapper, '_create_point') as mock_create:
            mock_point = Mock()
            mock_create.return_value = mock_point
            
            client_wrapper.write_market_data("BTCUSDT", data)
            
            # All None values should become 0.0
            call_args = mock_create.call_args[0]
            fields = call_args[2]
            for field_value in fields.values():
                assert field_value == 0.0

    def test_write_market_data_batch_mixed_types(self, client_wrapper):
        """Test writing market data batch with mixed data types."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        data_list = [
            {
                "symbol": "BTCUSDT",
                "price": Decimal("50000.00"),    # Decimal
                "volume": 100                    # int
            },
            {
                "symbol": "ETHUSD", 
                "price": 3000.0,                # float
                "volume": "200"                  # string
            }
        ]
        
        client_wrapper.write_market_data_batch(data_list)
        
        mock_write_api.write.assert_called_once()
        # Should handle all types without error

    def test_create_point_tags_conversion(self, client_wrapper):
        """Test that tags are properly converted to strings."""
        tags = {"symbol": "BTCUSDT", "exchange": "binance", "id": 123}
        fields = {"price": 50000.0}
        
        # Tags should be converted to strings in InfluxDB
        point = client_wrapper._create_point("test", tags, fields)
        assert isinstance(point, Point)


class TestInfluxDBClientWrapperErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def client_wrapper(self, mock_config=None):
        """Create InfluxDBClientWrapper instance for testing."""
        config = mock_config or Mock(spec=Config)
        return InfluxDBClientWrapper(
            url="http://localhost:8086",
            token="test_token", 
            org="test_org",
            bucket="test_bucket",
            config=config
        )

    def test_write_point_no_write_api(self, client_wrapper):
        """Test writing point when write_api is not initialized."""
        client_wrapper.write_api = None
        
        point = Point("test").field("value", 1.0)
        
        with pytest.raises(AttributeError):
            client_wrapper.write_point(point)

    def test_write_market_data_conversion_errors(self, client_wrapper):
        """Test market data writing with conversion errors."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        # Data that might cause conversion issues
        data = {
            "price": "invalid_number",  # Should be handled by _decimal_to_float
            "volume": float('inf'),     # Infinity value
            "bid": float('-inf'),       # Negative infinity
        }
        
        # Should handle gracefully without raising exception
        client_wrapper.write_market_data("BTCUSDT", data)
        
        mock_write_api.write.assert_called_once()

    def test_influxdb_connection_timeout(self, client_wrapper):
        """Test handling of connection timeout."""
        error = InfluxDBError("Connection timeout")
        
        with patch('src.database.influxdb_client.InfluxDBClient', side_effect=error):
            with patch.object(client_wrapper.error_handler, 'handle_error',
                            return_value=AsyncMock(return_value=False)) as mock_handle:
                with pytest.raises(DataSourceError):
                    import asyncio
                    asyncio.run(client_wrapper.connect())
                
                mock_handle.assert_called_once()

    def test_point_creation_edge_cases(self, client_wrapper):
        """Test point creation with edge case values."""
        tags = {"symbol": "TEST"}
        fields = {
            "zero": 0,
            "negative": -1.5,
            "large_number": 1e20,
            "small_number": 1e-20,
            "empty_string": "",
            "unicode": "测试",  # Unicode characters
        }
        
        point = client_wrapper._create_point("test", tags, fields)
        assert isinstance(point, Point)


class TestInfluxDBClientWrapperIntegration:
    """Integration-style tests for InfluxDB client wrapper."""

    @pytest.fixture
    def client_wrapper(self):
        """Create InfluxDBClientWrapper instance for testing."""
        return InfluxDBClientWrapper(
            url="http://localhost:8086",
            token="test_token",
            org="test_org",
            bucket="test_bucket"
        )

    def test_full_workflow_simulation(self, client_wrapper):
        """Test a complete workflow from connect to write."""
        mock_client = Mock()
        mock_write_api = Mock()
        mock_query_api = Mock()
        
        mock_client.write_api.return_value = mock_write_api
        mock_client.query_api.return_value = mock_query_api
        mock_client.ping.return_value = True
        
        with patch('src.database.influxdb_client.InfluxDBClient', return_value=mock_client):
            # Connect
            import asyncio
            asyncio.run(client_wrapper.connect())
            
            # Write market data
            data = {"price": Decimal("50000.00"), "volume": 100.0}
            client_wrapper.write_market_data("BTCUSDT", data)
            
            # Write batch data
            batch_data = [
                {"symbol": "BTCUSDT", "price": 50000.0},
                {"symbol": "ETHUSD", "price": 3000.0}
            ]
            client_wrapper.write_market_data_batch(batch_data)
            
            # Disconnect
            client_wrapper.disconnect()
            
            # Verify all operations were called
            mock_client.ping.assert_called()
            assert mock_write_api.write.call_count == 2  # Two write operations
            mock_client.close.assert_called_once()

    def test_connection_recovery_scenario(self, client_wrapper):
        """Test connection recovery after failure."""
        # First connection fails
        error = InfluxDBError("Initial connection failed")
        
        # Second connection succeeds
        mock_client = Mock()
        mock_client.ping.return_value = True
        
        with patch('src.database.influxdb_client.InfluxDBClient', 
                  side_effect=[error, mock_client]) as mock_influx:
            
            # First attempt should fail
            with pytest.raises(DataSourceError):
                import asyncio
                asyncio.run(client_wrapper.connect())
            
            # Second attempt should succeed
            import asyncio
            asyncio.run(client_wrapper.connect())
            
            # Should have been called twice
            assert mock_influx.call_count == 2
            assert client_wrapper.client == mock_client

    def test_concurrent_writes_simulation(self, client_wrapper):
        """Test handling of concurrent write operations."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        # Simulate multiple concurrent writes
        data1 = {"symbol": "BTCUSDT", "price": 50000.0}
        data2 = {"symbol": "ETHUSD", "price": 3000.0}
        data3 = {"symbol": "ADAUSDT", "price": 1.0}
        
        client_wrapper.write_market_data("BTCUSDT", data1)
        client_wrapper.write_market_data("ETHUSD", data2) 
        client_wrapper.write_market_data("ADAUSDT", data3)
        
        # Should handle all writes
        assert mock_write_api.write.call_count == 3

    def test_large_batch_handling(self, client_wrapper):
        """Test handling of large data batches."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        # Create large batch
        large_batch = []
        for i in range(1000):
            large_batch.append({
                "symbol": f"TEST{i:04d}",
                "price": 1000.0 + i,
                "volume": 100.0
            })
        
        client_wrapper.write_market_data_batch(large_batch)
        
        mock_write_api.write.assert_called_once()
        call_args = mock_write_api.write.call_args
        points = call_args[1]["record"]
        assert len(points) == 1000


class TestInfluxDBClientWrapperPerformance:
    """Performance-related tests for InfluxDB client."""

    @pytest.fixture
    def client_wrapper(self):
        """Create InfluxDBClientWrapper instance for testing."""
        return InfluxDBClientWrapper(
            url="http://localhost:8086",
            token="test_token",
            org="test_org",
            bucket="test_bucket"
        )

    def test_decimal_conversion_performance(self, client_wrapper):
        """Test Decimal conversion performance with various inputs."""
        # Test various numeric types
        test_values = [
            Decimal("123.456"),
            123.456,
            123,
            "123.456",
            None,
            0,
            Decimal("0.00000001"),  # Very small
            Decimal("99999999999999.99999999"),  # Very large
        ]
        
        for value in test_values:
            result = client_wrapper._decimal_to_float(value)
            assert isinstance(result, float)

    def test_point_creation_efficiency(self, client_wrapper):
        """Test point creation with various field combinations."""
        base_tags = {"symbol": "BTCUSDT", "exchange": "binance"}
        
        # Test different field combinations
        field_combinations = [
            {"price": 50000.0},  # Single field
            {"price": 50000.0, "volume": 100.0},  # Two fields
            {  # Many fields
                "price": 50000.0,
                "volume": 100.0,
                "bid": 49950.0,
                "ask": 50050.0,
                "open": 49000.0,
                "high": 51000.0,
                "low": 48000.0,
                "count": 1000,
                "active": True,
                "status": "trading"
            }
        ]
        
        for fields in field_combinations:
            point = client_wrapper._create_point("market_data", base_tags, fields)
            assert isinstance(point, Point)

    def test_batch_size_optimization(self, client_wrapper):
        """Test batch writing with different sizes."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        # Test different batch sizes
        batch_sizes = [1, 10, 100, 1000]
        
        for size in batch_sizes:
            batch = []
            for i in range(size):
                batch.append({
                    "symbol": f"TEST{i}",
                    "price": 1000.0 + i,
                    "volume": 100.0
                })
            
            client_wrapper.write_market_data_batch(batch)
            
            # Verify batch was processed
            call_args = mock_write_api.write.call_args
            points = call_args[1]["record"]
            assert len(points) == size
            
            mock_write_api.reset_mock()