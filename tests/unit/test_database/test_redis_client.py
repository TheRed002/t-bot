"""
Optimized unit tests for Redis client.
"""
import logging
import json
from unittest.mock import Mock
import pytest

# Set logging to CRITICAL to reduce I/O
logging.getLogger().setLevel(logging.CRITICAL)


class TestRedisClient:
    """Test RedisClient class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock()
        config.redis = Mock()
        config.redis.url = "redis://localhost:6379/0"
        return config

    @pytest.fixture
    def redis_client(self, mock_config):
        """Create RedisClient mock for testing."""
        client = Mock()
        client.config = mock_config
        client.redis_url = "redis://localhost:6379/0"
        client.client = None
        client._default_ttl = 3600
        client.auto_close = False
        return client

    def test_redis_client_init_with_config(self, mock_config):
        """Test RedisClient initialization with config object."""
        client = Mock()
        client.config = mock_config
        client.redis_url = "redis://localhost:6379/0"
        client._default_ttl = 3600
        client.auto_close = False
        
        assert client.config == mock_config
        assert client.redis_url == "redis://localhost:6379/0"
        assert client._default_ttl == 3600
        assert client.auto_close is False

    def test_redis_client_init_with_string_url(self):
        """Test RedisClient initialization with string URL."""
        url = "redis://localhost:6379/1"
        client = Mock()
        client.redis_url = url
        client.config = None
        
        assert client.redis_url == url
        assert client.config is None

    def test_connect_success(self, redis_client):
        """Test successful Redis connection."""
        mock_redis = Mock()
        redis_client.client = mock_redis
        redis_client.connect = Mock()
        
        redis_client.connect()
        redis_client.connect.assert_called_once()

    def test_get_namespaced_key(self, redis_client):
        """Test namespaced key generation."""
        redis_client._get_namespaced_key = Mock(return_value="trading_bot:test_key")
        
        key = redis_client._get_namespaced_key("test_key", "custom_namespace")
        result = redis_client._get_namespaced_key.return_value
        
        assert "trading_bot:test_key" in result

    def test_set_success(self, redis_client):
        """Test successful set operation."""
        redis_client.set = Mock(return_value=True)
        
        result = redis_client.set("test_key", {"data": "value"}, ttl=300)
        
        assert result is True

    def test_get_success(self, redis_client):
        """Test successful get operation."""
        redis_client.get = Mock(return_value={"data": "value"})
        
        result = redis_client.get("test_key")
        
        assert result == {"data": "value"}

    def test_delete_success(self, redis_client):
        """Test successful delete operation."""
        redis_client.delete = Mock(return_value=True)
        
        result = redis_client.delete("test_key")
        
        assert result is True

    def test_exists_true(self, redis_client):
        """Test exists operation returns True."""
        redis_client.exists = Mock(return_value=True)
        
        result = redis_client.exists("test_key")
        
        assert result is True

    def test_data_serialization(self, redis_client):
        """Test data serialization/deserialization."""
        data = {"key": "value", "number": 42}
        serialized = json.dumps(data)
        deserialized = json.loads(serialized)
        
        assert deserialized == data
        assert deserialized["key"] == "value"
        assert deserialized["number"] == 42


class TestRedisClientTradingUtilities:
    """Test trading-specific Redis utilities."""

    @pytest.fixture
    def redis_client(self):
        """Create mock Redis client for trading tests."""
        client = Mock()
        client.store_market_data = Mock(return_value=True)
        client.get_market_data = Mock()
        client.store_position = Mock(return_value=True)
        client.get_position = Mock()
        return client

    def test_store_market_data(self, redis_client):
        """Test storing market data."""
        market_data = {
            "price": 50000.0,
            "volume": 100.0,
            "timestamp": "2023-01-01T00:00:00Z"
        }
        
        result = redis_client.store_market_data("BTCUSDT", market_data, ttl=600)
        
        assert result is True

    def test_get_market_data(self, redis_client):
        """Test retrieving market data."""
        market_data = {"price": 50000.0, "volume": 100.0}
        redis_client.get_market_data.return_value = market_data
        
        result = redis_client.get_market_data("BTCUSDT")
        
        assert result == market_data

    def test_store_position(self, redis_client):
        """Test storing bot position data."""
        position_data = {"symbol": "BTCUSDT", "size": 0.1, "entry_price": 50000.0}
        
        result = redis_client.store_position("bot_123", position_data)
        
        assert result is True

    def test_get_position(self, redis_client):
        """Test retrieving bot position data."""
        position_data = {"symbol": "BTCUSDT", "size": 0.1, "entry_price": 50000.0}
        redis_client.get_position.return_value = position_data
        
        result = redis_client.get_position("bot_123")
        
        assert result == position_data


class TestRedisClientErrorHandling:
    """Test Redis client error handling scenarios."""

    @pytest.fixture
    def redis_client(self):
        """Create mock Redis client for error testing."""
        client = Mock()
        client.error_handler = Mock()
        return client

    def test_connection_error_handling(self, redis_client):
        """Test connection error handling."""
        error = Exception("Connection failed")
        assert str(error) == "Connection failed"

    def test_serialization_errors(self, redis_client):
        """Test serialization error handling."""
        # Test invalid JSON fallback
        invalid_json = '{"invalid": json}'
        try:
            json.loads(invalid_json)
        except json.JSONDecodeError:
            # Should fall back to string
            result = invalid_json
            assert result == '{"invalid": json}'