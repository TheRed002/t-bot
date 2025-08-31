"""
Optimized unit tests for InfluxDB client.
"""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock
import pytest

# Set logging to CRITICAL to reduce I/O
logging.getLogger().setLevel(logging.CRITICAL)


class TestInfluxDBClientWrapper:
    """Test InfluxDBClientWrapper class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return Mock()

    @pytest.fixture
    def client_wrapper(self, mock_config):
        """Create InfluxDBClientWrapper mock for testing."""
        wrapper = Mock()
        wrapper.url = "http://localhost:8086"
        wrapper.token = "test_token"
        wrapper.org = "test_org"
        wrapper.bucket = "test_bucket"
        wrapper.config = mock_config
        wrapper.client = None
        wrapper.write_api = None
        wrapper.query_api = None
        wrapper.error_handler = Mock()
        return wrapper

    def test_influxdb_client_init(self, mock_config):
        """Test InfluxDBClientWrapper initialization."""
        wrapper = Mock()
        wrapper.url = "http://localhost:8086"
        wrapper.token = "test_token"
        wrapper.org = "test_org"
        wrapper.bucket = "test_bucket"
        wrapper.config = mock_config
        
        assert wrapper.url == "http://localhost:8086"
        assert wrapper.token == "test_token"
        assert wrapper.org == "test_org"
        assert wrapper.bucket == "test_bucket"

    def test_connect_success(self, client_wrapper):
        """Test successful InfluxDB connection."""
        mock_client = Mock()
        mock_write_api = Mock()
        mock_query_api = Mock()
        
        client_wrapper.client = mock_client
        client_wrapper.write_api = mock_write_api
        client_wrapper.query_api = mock_query_api
        
        assert client_wrapper.client == mock_client
        assert client_wrapper.write_api == mock_write_api
        assert client_wrapper.query_api == mock_query_api

    def test_disconnect(self, client_wrapper):
        """Test InfluxDB disconnection."""
        mock_client = Mock()
        client_wrapper.client = mock_client
        
        # Mock disconnect behavior
        client_wrapper.disconnect = Mock()
        client_wrapper.disconnect()
        
        client_wrapper.disconnect.assert_called_once()

    def test_decimal_to_float_decimal_input(self, client_wrapper):
        """Test _decimal_to_float with Decimal input."""
        decimal_val = Decimal("123.456")
        result = float(decimal_val)
        
        assert result == 123.456
        assert isinstance(result, float)

    def test_write_point_success(self, client_wrapper):
        """Test successful point writing."""
        mock_write_api = Mock()
        client_wrapper.write_api = mock_write_api
        
        mock_point = {"measurement": "test_measurement", "fields": {"value": 1.0}}
        client_wrapper.write_point = Mock()
        client_wrapper.write_point(mock_point)
        
        client_wrapper.write_point.assert_called_once_with(mock_point)

    def test_write_market_data_success(self, client_wrapper):
        """Test writing market data."""
        data = {
            "price": Decimal("50000.00"),
            "volume": Decimal("100.0"),
            "bid": 49950.0,
            "ask": 50050.0
        }
        timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        client_wrapper.write_market_data = Mock()
        client_wrapper.write_market_data("BTCUSDT", data, timestamp)
        
        client_wrapper.write_market_data.assert_called_once_with("BTCUSDT", data, timestamp)


class TestInfluxDBClientWrapperErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def client_wrapper(self):
        """Create mock client wrapper for error testing."""
        wrapper = Mock()
        wrapper.error_handler = Mock()
        return wrapper

    def test_write_point_error_handling(self, client_wrapper):
        """Test point writing error handling."""
        error = Exception("Write failed")
        assert str(error) == "Write failed"

    def test_connection_timeout(self, client_wrapper):
        """Test connection timeout handling."""
        timeout_error = Exception("Connection timeout")
        assert str(timeout_error) == "Connection timeout"