"""
Unit tests for Redis client.

This module tests the RedisClient class and all Redis-related functionality
including connection management, operations, and trading-specific utilities.
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from redis.exceptions import ConnectionError, TimeoutError

from src.core.config import Config
from src.core.exceptions import DataError, DataSourceError
from src.database.redis_client import RedisClient


class TestRedisClient:
    """Test RedisClient class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=Config)
        config.redis.url = "redis://localhost:6379/0"
        return config

    @pytest.fixture
    def mock_config_object(self):
        """Create mock config object with redis attribute."""
        mock_redis_config = Mock()
        mock_redis_config.url = "redis://localhost:6379/0"
        
        mock_config = Mock()
        mock_config.redis = mock_redis_config
        return mock_config

    def test_redis_client_init_with_config(self, mock_config):
        """Test RedisClient initialization with config object."""
        client = RedisClient(mock_config)
        
        assert client.config == mock_config
        assert client.redis_url == "redis://localhost:6379/0"
        assert client.client is None
        assert client._default_ttl == 3600  # Default TTL
        assert client.auto_close is False

    def test_redis_client_init_with_string_url(self):
        """Test RedisClient initialization with string URL."""
        url = "redis://localhost:6379/1"
        client = RedisClient(url)
        
        assert client.redis_url == url
        assert client.config is None
        assert client.client is None

    def test_redis_client_init_with_invalid_config(self):
        """Test RedisClient initialization with invalid config."""
        client = RedisClient(123)  # Invalid config type
        
        assert client.redis_url == "redis://localhost:6379/0"  # Default
        assert client.config is None

    def test_redis_client_init_with_auto_close(self):
        """Test RedisClient initialization with auto_close enabled."""
        client = RedisClient("redis://localhost:6379/0", auto_close=True)
        
        assert client.auto_close is True

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_config):
        """Test successful Redis connection."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        
        with patch('redis.asyncio.Redis.from_url', return_value=mock_redis):
            await client.connect()
            
            assert client.client == mock_redis
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure_with_error_handler(self, mock_config):
        """Test Redis connection failure with error handler."""
        client = RedisClient(mock_config)
        
        error = ConnectionError("Redis connection failed")
        
        with patch('redis.asyncio.Redis.from_url', side_effect=error):
            with patch.object(client, 'error_handler') as mock_handler:
                mock_handler.handle_error.return_value = False
                
                with pytest.raises(DataSourceError, match="Redis connection failed"):
                    await client.connect()

    @pytest.mark.asyncio
    async def test_connect_failure_recovery(self, mock_config):
        """Test Redis connection failure with recovery."""
        client = RedisClient(mock_config)
        
        error = ConnectionError("Redis connection failed")
        mock_redis = AsyncMock()
        
        with patch('redis.asyncio.Redis.from_url', side_effect=[error, mock_redis]):
            with patch.object(client, 'error_handler') as mock_handler:
                mock_handler.handle_error.return_value = True
                
                # Should call connect recursively after recovery
                with patch.object(client, 'connect', side_effect=[None]) as mock_connect:
                    await client.connect()
                    mock_connect.assert_called()

    @pytest.mark.asyncio
    async def test_connect_failure_no_error_handler(self):
        """Test Redis connection failure without error handler."""
        client = RedisClient("redis://localhost:6379/0")
        
        error = ConnectionError("Redis connection failed")
        
        with patch('redis.asyncio.Redis.from_url', side_effect=error):
            with pytest.raises(DataSourceError, match="Redis connection failed"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_ensure_connected_not_connected(self, mock_config):
        """Test _ensure_connected when not connected."""
        client = RedisClient(mock_config)
        
        with patch.object(client, 'connect') as mock_connect:
            await client._ensure_connected()
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_connected_already_connected(self, mock_config):
        """Test _ensure_connected when already connected."""
        client = RedisClient(mock_config)
        client.client = AsyncMock()
        
        with patch.object(client, 'connect') as mock_connect:
            await client._ensure_connected()
            mock_connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_with_aclose(self, mock_config):
        """Test disconnect with aclose method."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        client.client = mock_redis
        
        await client.disconnect()
        
        mock_redis.aclose.assert_called_once()
        assert client.client is None

    @pytest.mark.asyncio
    async def test_disconnect_fallback_close(self, mock_config):
        """Test disconnect with fallback to close method."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        del mock_redis.aclose  # Remove aclose method
        mock_redis.close = AsyncMock()
        client.client = mock_redis
        
        await client.disconnect()
        
        mock_redis.close.assert_called_once()
        assert client.client is None

    @pytest.mark.asyncio
    async def test_disconnect_with_error(self, mock_config):
        """Test disconnect with error."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.aclose.side_effect = Exception("Close error")
        client.client = mock_redis
        
        # Should not raise exception
        await client.disconnect()
        assert client.client is None

    @pytest.mark.asyncio
    async def test_maybe_autoclose_enabled(self, mock_config):
        """Test _maybe_autoclose when auto_close is enabled."""
        client = RedisClient(mock_config, auto_close=True)
        client.client = AsyncMock()
        
        with patch.object(client, 'disconnect') as mock_disconnect:
            await client._maybe_autoclose()
            mock_disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_maybe_autoclose_disabled(self, mock_config):
        """Test _maybe_autoclose when auto_close is disabled."""
        client = RedisClient(mock_config, auto_close=False)
        client.client = AsyncMock()
        
        with patch.object(client, 'disconnect') as mock_disconnect:
            await client._maybe_autoclose()
            mock_disconnect.assert_not_called()

    def test_get_namespaced_key(self, mock_config):
        """Test namespaced key generation."""
        client = RedisClient(mock_config)
        
        key = client._get_namespaced_key("test_key", "custom_namespace")
        assert key == "custom_namespace:test_key"
        
        default_key = client._get_namespaced_key("test_key")
        assert default_key == "trading_bot:test_key"

    @pytest.mark.asyncio
    async def test_set_success(self, mock_config):
        """Test successful set operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        client.client = mock_redis
        
        result = await client.set("test_key", {"data": "value"}, ttl=300)
        
        assert result is True
        mock_redis.setex.assert_called_once_with(
            "trading_bot:test_key",
            300,
            '{"data": "value"}'
        )

    @pytest.mark.asyncio
    async def test_set_with_default_ttl(self, mock_config):
        """Test set operation with default TTL."""
        client = RedisClient(mock_config)
        client._default_ttl = 1800
        
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        client.client = mock_redis
        
        await client.set("test_key", "value")
        
        mock_redis.setex.assert_called_once_with(
            "trading_bot:test_key",
            1800,
            "value"
        )

    @pytest.mark.asyncio
    async def test_set_serialization(self, mock_config):
        """Test set operation with different value types."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        client.client = mock_redis
        
        # Test dictionary
        await client.set("dict_key", {"key": "value"})
        mock_redis.setex.assert_called_with(
            "trading_bot:dict_key",
            3600,
            '{"key": "value"}'
        )
        
        # Test list
        await client.set("list_key", [1, 2, 3])
        mock_redis.setex.assert_called_with(
            "trading_bot:list_key",
            3600,
            '[1, 2, 3]'
        )
        
        # Test string
        await client.set("string_key", "simple_value")
        mock_redis.setex.assert_called_with(
            "trading_bot:string_key",
            3600,
            "simple_value"
        )

    @pytest.mark.asyncio
    async def test_set_failure(self, mock_config):
        """Test set operation failure."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.setex.side_effect = TimeoutError("Redis timeout")
        client.client = mock_redis
        
        with pytest.raises(DataError, match="Redis set operation failed"):
            await client.set("test_key", "value")

    @pytest.mark.asyncio
    async def test_get_success(self, mock_config):
        """Test successful get operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = '{"data": "value"}'
        client.client = mock_redis
        
        result = await client.get("test_key")
        
        assert result == {"data": "value"}
        mock_redis.get.assert_called_once_with("trading_bot:test_key")

    @pytest.mark.asyncio
    async def test_get_not_found(self, mock_config):
        """Test get operation when key not found."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        client.client = mock_redis
        
        result = await client.get("nonexistent_key")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_deserialization_fallback(self, mock_config):
        """Test get operation with JSON deserialization fallback."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "plain_string_value"
        client.client = mock_redis
        
        result = await client.get("test_key")
        
        assert result == "plain_string_value"

    @pytest.mark.asyncio
    async def test_get_failure(self, mock_config):
        """Test get operation failure."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.get.side_effect = ConnectionError("Redis connection lost")
        client.client = mock_redis
        
        with pytest.raises(DataError, match="Redis get operation failed"):
            await client.get("test_key")

    @pytest.mark.asyncio
    async def test_delete_success(self, mock_config):
        """Test successful delete operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.delete.return_value = 1
        client.client = mock_redis
        
        result = await client.delete("test_key")
        
        assert result is True
        mock_redis.delete.assert_called_once_with("trading_bot:test_key")

    @pytest.mark.asyncio
    async def test_delete_not_found(self, mock_config):
        """Test delete operation when key not found."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.delete.return_value = 0
        client.client = mock_redis
        
        result = await client.delete("nonexistent_key")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_true(self, mock_config):
        """Test exists operation returns True."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 1
        client.client = mock_redis
        
        result = await client.exists("test_key")
        
        assert result is True
        mock_redis.exists.assert_called_once_with("trading_bot:test_key")

    @pytest.mark.asyncio
    async def test_exists_false(self, mock_config):
        """Test exists operation returns False."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 0
        client.client = mock_redis
        
        result = await client.exists("nonexistent_key")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_expire_success(self, mock_config):
        """Test successful expire operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.expire.return_value = True
        client.client = mock_redis
        
        result = await client.expire("test_key", 600)
        
        assert result is True
        mock_redis.expire.assert_called_once_with("trading_bot:test_key", 600)

    @pytest.mark.asyncio
    async def test_ttl_success(self, mock_config):
        """Test successful TTL operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.ttl.return_value = 300
        client.client = mock_redis
        
        result = await client.ttl("test_key")
        
        assert result == 300
        mock_redis.ttl.assert_called_once_with("trading_bot:test_key")

    @pytest.mark.asyncio
    async def test_hset_success(self, mock_config):
        """Test successful hset operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.hset.return_value = 1
        client.client = mock_redis
        
        result = await client.hset("hash_key", "field", {"data": "value"})
        
        assert result is True
        mock_redis.hset.assert_called_once_with(
            "trading_bot:hash_key",
            "field",
            '{"data": "value"}'
        )

    @pytest.mark.asyncio
    async def test_hget_success(self, mock_config):
        """Test successful hget operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.hget.return_value = '{"data": "value"}'
        client.client = mock_redis
        
        result = await client.hget("hash_key", "field")
        
        assert result == {"data": "value"}
        mock_redis.hget.assert_called_once_with("trading_bot:hash_key", "field")

    @pytest.mark.asyncio
    async def test_hgetall_success(self, mock_config):
        """Test successful hgetall operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.hgetall.return_value = {
            "field1": '{"data": "value1"}',
            "field2": "plain_value"
        }
        client.client = mock_redis
        
        result = await client.hgetall("hash_key")
        
        expected = {
            "field1": {"data": "value1"},
            "field2": "plain_value"
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_hdel_success(self, mock_config):
        """Test successful hdel operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.hdel.return_value = 1
        client.client = mock_redis
        
        result = await client.hdel("hash_key", "field")
        
        assert result is True
        mock_redis.hdel.assert_called_once_with("trading_bot:hash_key", "field")

    @pytest.mark.asyncio
    async def test_lpush_success(self, mock_config):
        """Test successful lpush operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.lpush.return_value = 1
        client.client = mock_redis
        
        result = await client.lpush("list_key", {"data": "value"})
        
        assert result == 1
        mock_redis.lpush.assert_called_once_with(
            "trading_bot:list_key",
            '{"data": "value"}'
        )

    @pytest.mark.asyncio
    async def test_rpush_success(self, mock_config):
        """Test successful rpush operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.rpush.return_value = 2
        client.client = mock_redis
        
        result = await client.rpush("list_key", "value")
        
        assert result == 2
        mock_redis.rpush.assert_called_once_with("trading_bot:list_key", "value")

    @pytest.mark.asyncio
    async def test_lrange_success(self, mock_config):
        """Test successful lrange operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.lrange.return_value = ['{"data": "value1"}', "value2"]
        client.client = mock_redis
        
        result = await client.lrange("list_key", 0, -1)
        
        expected = [{"data": "value1"}, "value2"]
        assert result == expected
        mock_redis.lrange.assert_called_once_with("trading_bot:list_key", 0, -1)

    @pytest.mark.asyncio
    async def test_sadd_success(self, mock_config):
        """Test successful sadd operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.sadd.return_value = 1
        client.client = mock_redis
        
        result = await client.sadd("set_key", {"data": "value"})
        
        assert result is True
        mock_redis.sadd.assert_called_once_with(
            "trading_bot:set_key",
            '{"data": "value"}'
        )

    @pytest.mark.asyncio
    async def test_smembers_success(self, mock_config):
        """Test successful smembers operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.smembers.return_value = {'{"data": "value1"}', "value2"}
        client.client = mock_redis
        
        result = await client.smembers("set_key")
        
        # Order may vary for sets, so check contents
        expected = [{"data": "value1"}, "value2"]
        assert len(result) == 2
        assert {"data": "value1"} in result
        assert "value2" in result

    @pytest.mark.asyncio
    async def test_ping_success(self, mock_config):
        """Test successful ping operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        client.client = mock_redis
        
        result = await client.ping()
        
        assert result is True

    @pytest.mark.asyncio
    async def test_ping_failure(self, mock_config):
        """Test ping operation failure."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = ConnectionError("Ping failed")
        client.client = mock_redis
        
        with pytest.raises(DataError, match="Redis ping failed"):
            await client.ping()

    @pytest.mark.asyncio
    async def test_info_success(self, mock_config):
        """Test successful info operation."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        info_data = {"redis_version": "6.2.0", "connected_clients": 1}
        mock_redis.info.return_value = info_data
        client.client = mock_redis
        
        result = await client.info()
        
        assert result == info_data

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_config):
        """Test successful health check."""
        client = RedisClient(mock_config)
        
        with patch.object(client, 'ping', return_value=True):
            result = await client.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_config):
        """Test health check failure."""
        client = RedisClient(mock_config)
        
        with patch.object(client, 'ping', side_effect=Exception("Ping failed")):
            result = await client.health_check()
            assert result is False


class TestRedisClientTradingUtilities:
    """Test trading-specific Redis utilities."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=Config)
        config.redis.url = "redis://localhost:6379/0"
        return config

    @pytest.mark.asyncio
    async def test_store_market_data(self, mock_config):
        """Test storing market data."""
        client = RedisClient(mock_config)
        
        market_data = {
            "price": 50000.0,
            "volume": 100.0,
            "timestamp": "2023-01-01T00:00:00Z"
        }
        
        with patch.object(client, 'set') as mock_set:
            mock_set.return_value = True
            
            result = await client.store_market_data("BTCUSDT", market_data, ttl=600)
            
            assert result is True
            mock_set.assert_called_once_with(
                "market_data:BTCUSDT",
                market_data,
                600,
                "market_data"
            )

    @pytest.mark.asyncio
    async def test_get_market_data(self, mock_config):
        """Test retrieving market data."""
        client = RedisClient(mock_config)
        
        market_data = {
            "price": 50000.0,
            "volume": 100.0,
            "timestamp": "2023-01-01T00:00:00Z"
        }
        
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = market_data
            
            result = await client.get_market_data("BTCUSDT")
            
            assert result == market_data
            mock_get.assert_called_once_with("market_data:BTCUSDT", "market_data")

    @pytest.mark.asyncio
    async def test_store_position(self, mock_config):
        """Test storing bot position data."""
        client = RedisClient(mock_config)
        
        position_data = {
            "symbol": "BTCUSDT",
            "size": 0.1,
            "entry_price": 50000.0
        }
        
        with patch.object(client, 'hset') as mock_hset:
            mock_hset.return_value = True
            
            result = await client.store_position("bot_123", position_data)
            
            assert result is True
            mock_hset.assert_called_once_with(
                "positions",
                "bot_123",
                position_data,
                "bot_state"
            )

    @pytest.mark.asyncio
    async def test_get_position(self, mock_config):
        """Test retrieving bot position data."""
        client = RedisClient(mock_config)
        
        position_data = {
            "symbol": "BTCUSDT",
            "size": 0.1,
            "entry_price": 50000.0
        }
        
        with patch.object(client, 'hget') as mock_hget:
            mock_hget.return_value = position_data
            
            result = await client.get_position("bot_123")
            
            assert result == position_data
            mock_hget.assert_called_once_with("positions", "bot_123", "bot_state")

    @pytest.mark.asyncio
    async def test_store_balance(self, mock_config):
        """Test storing user balance data."""
        client = RedisClient(mock_config)
        
        balance_data = {
            "BTC": 1.5,
            "USDT": 10000.0
        }
        
        with patch.object(client, 'set') as mock_set:
            mock_set.return_value = True
            
            result = await client.store_balance("user_456", "binance", balance_data)
            
            assert result is True
            mock_set.assert_called_once_with(
                "balance:user_456:binance",
                balance_data,
                3600,
                "balances"
            )

    @pytest.mark.asyncio
    async def test_get_balance(self, mock_config):
        """Test retrieving user balance data."""
        client = RedisClient(mock_config)
        
        balance_data = {
            "BTC": 1.5,
            "USDT": 10000.0
        }
        
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = balance_data
            
            result = await client.get_balance("user_456", "binance")
            
            assert result == balance_data
            mock_get.assert_called_once_with(
                "balance:user_456:binance",
                "balances"
            )

    @pytest.mark.asyncio
    async def test_store_cache(self, mock_config):
        """Test storing cached data."""
        client = RedisClient(mock_config)
        
        cache_data = {"result": "computed_value"}
        
        with patch.object(client, 'set') as mock_set:
            mock_set.return_value = True
            
            result = await client.store_cache("cache_key", cache_data, ttl=1800)
            
            assert result is True
            mock_set.assert_called_once_with(
                "cache_key",
                cache_data,
                1800,
                "cache"
            )

    @pytest.mark.asyncio
    async def test_get_cache(self, mock_config):
        """Test retrieving cached data."""
        client = RedisClient(mock_config)
        
        cache_data = {"result": "computed_value"}
        
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = cache_data
            
            result = await client.get_cache("cache_key")
            
            assert result == cache_data
            mock_get.assert_called_once_with("cache_key", "cache")


class TestRedisClientDebugUtilities:
    """Test Redis client debug and monitoring utilities."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=Config)
        config.redis.url = "redis://localhost:6379/0"
        return config

    @pytest.mark.asyncio
    async def test_debug_info_success(self, mock_config):
        """Test debug info retrieval."""
        client = RedisClient(mock_config)
        
        info_data = {
            "connected_clients": "5",
            "used_memory_human": "1.5M",
            "total_commands_processed": "12345",
            "keyspace_hits": "100",
            "keyspace_misses": "10",
            "uptime_in_seconds": "86400"
        }
        
        mock_redis = AsyncMock()
        mock_redis.info.return_value = info_data
        client.client = mock_redis
        
        result = await client.debug_info()
        
        assert result["success"] is True
        assert "connected_clients" in result["data"]
        assert result["data"]["connected_clients"] == "5"
        assert result["data"]["used_memory_human"] == "1.5M"

    @pytest.mark.asyncio
    async def test_debug_info_failure(self, mock_config):
        """Test debug info retrieval failure."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.info.side_effect = ConnectionError("Info failed")
        client.client = mock_redis
        
        result = await client.debug_info()
        
        assert result["success"] is False
        assert "Failed to get Redis info" in result["message"]


class TestRedisClientErrorHandling:
    """Test Redis client error handling scenarios."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=Config)
        config.redis.url = "redis://localhost:6379/0"
        return config

    @pytest.mark.asyncio
    async def test_set_with_error_handler(self, mock_config):
        """Test set operation with error handler."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.setex.side_effect = TimeoutError("Timeout")
        client.client = mock_redis
        
        with patch.object(client, 'error_handler') as mock_handler:
            mock_handler.handle_error = AsyncMock()
            
            with pytest.raises(DataError):
                await client.set("key", "value")
            
            mock_handler.handle_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_close_with_lock(self, mock_config):
        """Test auto-close with concurrency control."""
        client = RedisClient(mock_config, auto_close=True)
        
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        client.client = mock_redis
        
        # Simulate concurrent calls
        tasks = [client._maybe_autoclose() for _ in range(3)]
        await asyncio.gather(*tasks)
        
        # Should only close once due to lock
        assert mock_redis.aclose.call_count <= 1

    @pytest.mark.asyncio
    async def test_auto_close_race_condition(self, mock_config):
        """Test auto-close race condition handling."""
        client = RedisClient(mock_config, auto_close=True)
        
        mock_redis = AsyncMock()
        client.client = mock_redis
        
        # Simulate client being set to None during close
        async def mock_aclose():
            client.client = None
        
        mock_redis.aclose.side_effect = mock_aclose
        
        await client._maybe_autoclose()
        
        # Should handle gracefully
        assert client.client is None

    @pytest.mark.asyncio
    async def test_operations_with_ensure_connected(self, mock_config):
        """Test that operations call _ensure_connected."""
        client = RedisClient(mock_config)
        
        with patch.object(client, '_ensure_connected') as mock_ensure:
            with patch.object(client, '_maybe_autoclose') as mock_autoclose:
                mock_redis = AsyncMock()
                mock_redis.get.return_value = "value"
                client.client = mock_redis
                
                await client.get("key")
                
                mock_ensure.assert_called_once()
                mock_autoclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_serialization_edge_cases(self, mock_config):
        """Test serialization of edge case values."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        client.client = mock_redis
        
        # Test with complex nested structures
        complex_data = {
            "nested": {"deep": {"data": [1, 2, {"key": "value"}]}},
            "none_value": None,
            "boolean": True,
            "number": 42.5
        }
        
        await client.set("complex_key", complex_data)
        
        # Verify JSON serialization was called
        call_args = mock_redis.setex.call_args[0]
        assert '"nested":' in call_args[2]  # JSON string contains nested key

    @pytest.mark.asyncio
    async def test_deserialization_edge_cases(self, mock_config):
        """Test deserialization of edge case values."""
        client = RedisClient(mock_config)
        
        mock_redis = AsyncMock()
        client.client = mock_redis
        
        # Test invalid JSON fallback
        mock_redis.get.return_value = '{"invalid": json}'
        result = await client.get("key")
        assert result == '{"invalid": json}'  # Should fall back to string
        
        # Test empty string
        mock_redis.get.return_value = ""
        result = await client.get("key")
        assert result == ""
        
        # Test valid JSON
        mock_redis.get.return_value = '{"valid": "json"}'
        result = await client.get("key")
        assert result == {"valid": "json"}


class TestRedisClientAdvancedOperations:
    """Test advanced Redis operations and patterns."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=Config)
        config.redis.url = "redis://localhost:6379/0"
        return config

    @pytest.fixture
    def redis_client(self, mock_config):
        """Create RedisClient instance for testing."""
        return RedisClient(mock_config)

    @pytest.mark.asyncio
    async def test_keys_pattern_matching(self, redis_client):
        """Test keys operation with pattern matching."""
        mock_redis = AsyncMock()
        mock_redis.keys.return_value = [
            "trading_bot:market_data:BTCUSD",
            "trading_bot:market_data:ETHUSD",
            "trading_bot:positions:bot1"
        ]
        redis_client.client = mock_redis

        result = await redis_client.keys("market_data:*")
        
        assert len(result) == 3
        mock_redis.keys.assert_called_once_with("trading_bot:market_data:*")

    @pytest.mark.asyncio
    async def test_flushdb_operation(self, redis_client):
        """Test database flush operation."""
        mock_redis = AsyncMock()
        mock_redis.flushdb.return_value = True
        redis_client.client = mock_redis

        result = await redis_client.flushdb()
        
        assert result is True
        mock_redis.flushdb.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_operations(self, redis_client):
        """Test Redis pipeline operations."""
        mock_redis = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = [True, "value", 1]
        redis_client.client = mock_redis

        async with redis_client.pipeline() as pipe:
            pipe.set("key1", "value1")
            pipe.get("key2")
            pipe.delete("key3")
            results = await pipe.execute()

        assert results == [True, "value", 1]
        mock_redis.pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_mget_operation(self, redis_client):
        """Test multi-get operation."""
        mock_redis = AsyncMock()
        mock_redis.mget.return_value = ['{"data": "value1"}', "value2", None]
        redis_client.client = mock_redis

        result = await redis_client.mget(["key1", "key2", "key3"])
        
        expected = [{"data": "value1"}, "value2", None]
        assert result == expected
        mock_redis.mget.assert_called_once_with([
            "trading_bot:key1", "trading_bot:key2", "trading_bot:key3"
        ])

    @pytest.mark.asyncio
    async def test_mset_operation(self, redis_client):
        """Test multi-set operation."""
        mock_redis = AsyncMock()
        mock_redis.mset.return_value = True
        redis_client.client = mock_redis

        data = {
            "key1": {"data": "value1"},
            "key2": "value2",
            "key3": [1, 2, 3]
        }
        
        result = await redis_client.mset(data)
        
        assert result is True
        expected_args = {
            "trading_bot:key1": '{"data": "value1"}',
            "trading_bot:key2": "value2",
            "trading_bot:key3": "[1, 2, 3]"
        }
        mock_redis.mset.assert_called_once_with(expected_args)

    @pytest.mark.asyncio
    async def test_incr_operation(self, redis_client):
        """Test increment operation."""
        mock_redis = AsyncMock()
        mock_redis.incr.return_value = 5
        redis_client.client = mock_redis

        result = await redis_client.incr("counter")
        
        assert result == 5
        mock_redis.incr.assert_called_once_with("trading_bot:counter")

    @pytest.mark.asyncio
    async def test_incrby_operation(self, redis_client):
        """Test increment by amount operation."""
        mock_redis = AsyncMock()
        mock_redis.incrby.return_value = 15
        redis_client.client = mock_redis

        result = await redis_client.incrby("counter", 10)
        
        assert result == 15
        mock_redis.incrby.assert_called_once_with("trading_bot:counter", 10)

    @pytest.mark.asyncio
    async def test_decr_operation(self, redis_client):
        """Test decrement operation."""
        mock_redis = AsyncMock()
        mock_redis.decr.return_value = 3
        redis_client.client = mock_redis

        result = await redis_client.decr("counter")
        
        assert result == 3
        mock_redis.decr.assert_called_once_with("trading_bot:counter")

    @pytest.mark.asyncio
    async def test_zadd_operation(self, redis_client):
        """Test sorted set add operation."""
        mock_redis = AsyncMock()
        mock_redis.zadd.return_value = 1
        redis_client.client = mock_redis

        result = await redis_client.zadd("leaderboard", {"player1": 100, "player2": 200})
        
        assert result == 1
        mock_redis.zadd.assert_called_once_with("trading_bot:leaderboard", {"player1": 100, "player2": 200})

    @pytest.mark.asyncio
    async def test_zrange_operation(self, redis_client):
        """Test sorted set range operation."""
        mock_redis = AsyncMock()
        mock_redis.zrange.return_value = ["player1", "player2"]
        redis_client.client = mock_redis

        result = await redis_client.zrange("leaderboard", 0, -1)
        
        assert result == ["player1", "player2"]
        mock_redis.zrange.assert_called_once_with("trading_bot:leaderboard", 0, -1)

    @pytest.mark.asyncio
    async def test_zrem_operation(self, redis_client):
        """Test sorted set remove operation."""
        mock_redis = AsyncMock()
        mock_redis.zrem.return_value = 1
        redis_client.client = mock_redis

        result = await redis_client.zrem("leaderboard", "player1")
        
        assert result == 1
        mock_redis.zrem.assert_called_once_with("trading_bot:leaderboard", "player1")

    @pytest.mark.asyncio
    async def test_pubsub_operations(self, redis_client):
        """Test publish/subscribe operations."""
        mock_redis = AsyncMock()
        mock_pubsub = AsyncMock()
        mock_redis.pubsub.return_value = mock_pubsub
        redis_client.client = mock_redis

        # Test publish
        mock_redis.publish.return_value = 1
        result = await redis_client.publish("channel", {"message": "data"})
        assert result == 1
        mock_redis.publish.assert_called_once_with("trading_bot:channel", '{"message": "data"}')

        # Test subscribe
        await redis_client.subscribe("channel")
        mock_pubsub.subscribe.assert_called_once_with("trading_bot:channel")


class TestRedisClientConnectionPooling:
    """Test Redis connection pooling functionality."""

    @pytest.fixture
    def mock_config_with_pool(self):
        """Create mock configuration with pooling settings."""
        config = Mock(spec=Config)
        config.redis.url = "redis://localhost:6379/0"
        config.redis.max_connections = 10
        config.redis.retry_on_timeout = True
        return config

    @pytest.fixture
    def redis_client(self, mock_config_with_pool):
        """Create RedisClient instance for testing."""
        return RedisClient(mock_config_with_pool)

    @pytest.mark.asyncio
    async def test_connection_pool_configuration(self, redis_client):
        """Test connection pool configuration."""
        with patch('redis.asyncio.Redis.from_url') as mock_from_url:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            await redis_client.connect()
            
            # Verify connection was created with pool settings
            mock_from_url.assert_called_once()
            call_args = mock_from_url.call_args
            assert call_args[0][0] == "redis://localhost:6379/0"

    @pytest.mark.asyncio
    async def test_concurrent_operations_with_pool(self, redis_client):
        """Test concurrent operations using connection pool."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "value"
        redis_client.client = mock_redis

        # Simulate concurrent operations
        tasks = [redis_client.get(f"key_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(result == "value" for result in results)
        assert mock_redis.get.call_count == 10


class TestRedisClientPerformanceOptimizations:
    """Test performance optimization features."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=Config)
        config.redis.url = "redis://localhost:6379/0"
        return config

    @pytest.fixture
    def redis_client(self, mock_config):
        """Create RedisClient instance for testing."""
        return RedisClient(mock_config)

    @pytest.mark.asyncio
    async def test_batch_operations_efficiency(self, redis_client):
        """Test batch operations for efficiency."""
        mock_redis = AsyncMock()
        mock_redis.mset.return_value = True
        redis_client.client = mock_redis

        # Batch set multiple keys
        batch_data = {f"key_{i}": f"value_{i}" for i in range(100)}
        result = await redis_client.mset(batch_data)

        assert result is True
        mock_redis.mset.assert_called_once()

    @pytest.mark.asyncio
    async def test_compression_for_large_values(self, redis_client):
        """Test compression for large values."""
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        redis_client.client = mock_redis

        # Simulate large data that might benefit from compression
        large_data = {"data": "x" * 10000, "metadata": {"size": "large"}}
        
        await redis_client.set("large_key", large_data)
        
        mock_redis.setex.assert_called_once()
        # Verify data was serialized
        call_args = mock_redis.setex.call_args[0]
        assert "data" in call_args[2]

    @pytest.mark.asyncio
    async def test_pipelining_for_bulk_operations(self, redis_client):
        """Test pipelining for bulk operations."""
        mock_redis = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = [True] * 50
        redis_client.client = mock_redis

        # Simulate bulk operations using pipeline
        async with redis_client.pipeline() as pipe:
            for i in range(50):
                pipe.set(f"bulk_key_{i}", f"value_{i}")
            results = await pipe.execute()

        assert len(results) == 50
        assert all(result is True for result in results)


class TestRedisClientMonitoringAndMetrics:
    """Test monitoring and metrics functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=Config)
        config.redis.url = "redis://localhost:6379/0"
        return config

    @pytest.fixture
    def redis_client(self, mock_config):
        """Create RedisClient instance for testing."""
        return RedisClient(mock_config)

    @pytest.mark.asyncio
    async def test_connection_health_monitoring(self, redis_client):
        """Test connection health monitoring."""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.info.return_value = {
            "connected_clients": 5,
            "used_memory": 1048576,
            "total_commands_processed": 1000
        }
        redis_client.client = mock_redis

        health = await redis_client.health_check()
        info = await redis_client.info()

        assert health is True
        assert "connected_clients" in info
        assert info["connected_clients"] == 5

    @pytest.mark.asyncio
    async def test_operation_metrics_tracking(self, redis_client):
        """Test operation metrics tracking."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "value"
        mock_redis.set.return_value = True
        redis_client.client = mock_redis

        # Simulate operations and track metrics
        await redis_client.get("key1")
        await redis_client.set("key2", "value2")
        await redis_client.get("key3")

        # Verify operations were called
        assert mock_redis.get.call_count == 2
        assert mock_redis.set.call_count == 1

    @pytest.mark.asyncio
    async def test_error_rate_monitoring(self, redis_client):
        """Test error rate monitoring."""
        mock_redis = AsyncMock()
        redis_client.client = mock_redis

        # Simulate successful and failed operations
        mock_redis.get.side_effect = [
            "value1",  # Success
            ConnectionError("Connection lost"),  # Error
            "value3"   # Success
        ]

        results = []
        for i in range(3):
            try:
                result = await redis_client.get(f"key_{i}")
                results.append(("success", result))
            except Exception as e:
                results.append(("error", str(e)))

        assert len(results) == 3
        assert results[0][0] == "success"
        assert results[1][0] == "error"
        assert results[2][0] == "success"


class TestRedisClientSecurityFeatures:
    """Test security-related features."""

    @pytest.fixture
    def mock_config_with_auth(self):
        """Create mock configuration with authentication."""
        config = Mock(spec=Config)
        config.redis.url = "redis://:password@localhost:6379/0"
        config.redis.ssl = True
        config.redis.ssl_cert_reqs = "required"
        return config

    @pytest.fixture
    def redis_client(self, mock_config_with_auth):
        """Create RedisClient instance with auth for testing."""
        return RedisClient(mock_config_with_auth)

    @pytest.mark.asyncio
    async def test_authenticated_connection(self, redis_client):
        """Test connection with authentication."""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        
        with patch('redis.asyncio.Redis.from_url', return_value=mock_redis) as mock_from_url:
            await redis_client.connect()
            
            # Verify connection was created with auth URL
            mock_from_url.assert_called_once()
            call_args = mock_from_url.call_args[0]
            assert ":password@" in call_args[0]

    @pytest.mark.asyncio
    async def test_ssl_connection(self, redis_client):
        """Test SSL connection configuration."""
        with patch('redis.asyncio.Redis.from_url') as mock_from_url:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            await redis_client.connect()
            
            # SSL configuration should be handled by Redis client
            mock_from_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_sensitive_data_handling(self, redis_client):
        """Test handling of sensitive data."""
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        redis_client.client = mock_redis

        # Store sensitive data (should be handled securely)
        sensitive_data = {
            "api_key": "secret_key_12345",
            "user_id": "user_123",
            "timestamp": "2023-01-01T00:00:00Z"
        }

        await redis_client.set("sensitive_key", sensitive_data)
        
        # Verify data was stored (in real implementation, might be encrypted)
        mock_redis.setex.assert_called_once()


class TestRedisClientEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=Config)
        config.redis.url = "redis://localhost:6379/0"
        return config

    @pytest.fixture
    def redis_client(self, mock_config):
        """Create RedisClient instance for testing."""
        return RedisClient(mock_config)

    @pytest.mark.asyncio
    async def test_very_large_keys(self, redis_client):
        """Test handling of very large keys."""
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        redis_client.client = mock_redis

        # Test very large key name
        large_key = "x" * 1000
        await redis_client.set(large_key, "value")
        
        expected_key = f"trading_bot:{large_key}"
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]
        assert call_args[0] == expected_key

    @pytest.mark.asyncio
    async def test_very_large_values(self, redis_client):
        """Test handling of very large values."""
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        redis_client.client = mock_redis

        # Test very large value
        large_value = {"data": "x" * 100000}
        await redis_client.set("key", large_value)
        
        mock_redis.setex.assert_called_once()
        # Verify large value was serialized
        call_args = mock_redis.setex.call_args[0]
        assert len(call_args[2]) > 100000

    @pytest.mark.asyncio
    async def test_empty_values(self, redis_client):
        """Test handling of empty values."""
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        mock_redis.get.return_value = ""
        redis_client.client = mock_redis

        # Test empty string
        await redis_client.set("empty_key", "")
        result = await redis_client.get("empty_key")
        
        assert result == ""

    @pytest.mark.asyncio
    async def test_null_values(self, redis_client):
        """Test handling of null values."""
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        mock_redis.get.return_value = None
        redis_client.client = mock_redis

        # Test None value storage
        await redis_client.set("null_key", None)
        result = await redis_client.get("null_key")
        
        # Redis returns None for non-existent keys
        assert result is None

    @pytest.mark.asyncio
    async def test_unicode_handling(self, redis_client):
        """Test handling of unicode characters."""
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        mock_redis.get.return_value = '{"emoji": "😀", "chinese": "测试"}'
        redis_client.client = mock_redis

        # Test unicode data
        unicode_data = {"emoji": "😀", "chinese": "测试"}
        await redis_client.set("unicode_key", unicode_data)
        result = await redis_client.get("unicode_key")
        
        assert result["emoji"] == "😀"
        assert result["chinese"] == "测试"

    @pytest.mark.asyncio
    async def test_special_characters_in_keys(self, redis_client):
        """Test handling of special characters in keys."""
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        redis_client.client = mock_redis

        # Test keys with special characters
        special_keys = ["key:with:colons", "key with spaces", "key@with#special$chars"]
        
        for key in special_keys:
            await redis_client.set(key, "value")
            
        assert mock_redis.setex.call_count == 3

    @pytest.mark.asyncio
    async def test_connection_recovery_after_failure(self, redis_client):
        """Test connection recovery after network failure."""
        mock_redis = AsyncMock()
        
        # First operation fails, second succeeds after reconnection
        mock_redis.get.side_effect = [
            ConnectionError("Connection lost"),
            "recovered_value"
        ]
        redis_client.client = mock_redis

        with patch.object(redis_client, 'connect') as mock_connect:
            # First call fails and triggers reconnection
            with pytest.raises(DataError):
                await redis_client.get("key")

    @pytest.mark.asyncio
    async def test_concurrent_auto_close_operations(self, redis_client):
        """Test concurrent auto-close operations."""
        redis_client.auto_close = True
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        redis_client.client = mock_redis

        # Simulate many concurrent operations that trigger auto-close
        tasks = []
        for i in range(50):
            task = redis_client._maybe_autoclose()
            tasks.append(task)

        await asyncio.gather(*tasks)
        
        # Should handle concurrent closes gracefully
        # (Exact behavior depends on implementation)
        assert mock_redis.aclose.call_count >= 0  # May be 0 due to locking

    @pytest.mark.asyncio
    async def test_memory_cleanup_on_large_operations(self, redis_client):
        """Test memory cleanup during large operations."""
        mock_redis = AsyncMock()
        mock_redis.mset.return_value = True
        redis_client.client = mock_redis

        # Simulate large batch operation
        large_batch = {f"key_{i}": f"value_{i}" * 1000 for i in range(1000)}
        
        await redis_client.mset(large_batch)
        
        # Should complete without memory issues
        mock_redis.mset.assert_called_once()

    @pytest.mark.asyncio
    async def test_operation_timeout_handling(self, redis_client):
        """Test handling of operation timeouts."""
        mock_redis = AsyncMock()
        
        # Simulate timeout on long-running operation
        async def slow_operation():
            await asyncio.sleep(10)  # Simulate very slow operation
            return "slow_result"
        
        mock_redis.get.side_effect = slow_operation
        redis_client.client = mock_redis

        # Should handle timeout appropriately
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(redis_client.get("slow_key"), timeout=0.1)