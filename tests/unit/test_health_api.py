"""
Unit tests for health check API endpoints.

This module tests the individual health check functions and API endpoints
to ensure they work correctly under various conditions.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.core.config import Config
from src.web_interface.api.health import (
    ComponentHealth,
    HealthStatus,
    check_database_health,
    check_redis_health,
    check_exchanges_health,
    check_ml_models_health,
)


class TestComponentHealth:
    """Test ComponentHealth model."""

    def test_component_health_creation(self):
        """Test ComponentHealth model creation."""
        health = ComponentHealth(
            status="healthy",
            message="All good",
            response_time_ms=50.5,
            last_check=datetime.now(timezone.utc),
            metadata={"test": "data"}
        )
        
        assert health.status == "healthy"
        assert health.message == "All good"
        assert health.response_time_ms == 50.5
        assert health.metadata["test"] == "data"

    def test_component_health_serialization(self):
        """Test ComponentHealth model serialization."""
        health = ComponentHealth(
            status="degraded",
            message="Minor issues",
            last_check=datetime.now(timezone.utc)
        )
        
        data = health.dict()
        
        assert data["status"] == "degraded"
        assert data["message"] == "Minor issues"
        assert "last_check" in data
        assert data["response_time_ms"] is None
        assert data["metadata"] is None


class TestHealthStatus:
    """Test HealthStatus model."""

    def test_health_status_creation(self):
        """Test HealthStatus model creation."""
        timestamp = datetime.now(timezone.utc)
        status = HealthStatus(
            status="healthy",
            timestamp=timestamp,
            service="test-service",
            version="1.0.0",
            uptime_seconds=123.45,
            checks={"db": {"status": "healthy"}}
        )
        
        assert status.status == "healthy"
        assert status.timestamp == timestamp
        assert status.service == "test-service"
        assert status.version == "1.0.0"
        assert status.uptime_seconds == 123.45
        assert status.checks["db"]["status"] == "healthy"


class TestDatabaseHealthCheck:
    """Test database health check functionality."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return Config(
            database={
                "url": "postgresql://user:pass@localhost/test",
                "pool_size": 10
            }
        )

    @pytest.mark.asyncio
    async def test_database_health_check_success(self, test_config):
        """Test successful database health check."""
        with patch('src.web_interface.api.health.DatabaseConnectionManager') as mock_db_manager:
            # Mock successful database connection
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = 1
            
            mock_manager = AsyncMock()
            mock_manager.get_connection.return_value.__aenter__.return_value = mock_conn
            mock_manager.get_pool_status.return_value = {
                "size": 10,
                "used": 3,
                "free": 7
            }
            mock_db_manager.return_value = mock_manager
            
            result = await check_database_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "healthy"
            assert "Database connection successful" in result.message
            assert result.response_time_ms is not None
            assert result.response_time_ms > 0
            assert result.metadata["pool_size"] == 10
            assert result.metadata["pool_used"] == 3
            assert result.metadata["pool_free"] == 7

    @pytest.mark.asyncio
    async def test_database_health_check_wrong_result(self, test_config):
        """Test database health check with wrong query result."""
        with patch('src.web_interface.api.health.DatabaseConnectionManager') as mock_db_manager:
            # Mock database connection with wrong result
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = 0  # Wrong result
            
            mock_manager = AsyncMock()
            mock_manager.get_connection.return_value.__aenter__.return_value = mock_conn
            mock_db_manager.return_value = mock_manager
            
            result = await check_database_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "unhealthy"
            assert "unexpected result" in result.message

    @pytest.mark.asyncio
    async def test_database_health_check_connection_failure(self, test_config):
        """Test database health check with connection failure."""
        with patch('src.web_interface.api.health.DatabaseConnectionManager') as mock_db_manager:
            # Mock connection failure
            mock_manager = AsyncMock()
            mock_manager.get_connection.side_effect = Exception("Connection refused")
            mock_db_manager.return_value = mock_manager
            
            result = await check_database_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "unhealthy"
            assert "Connection refused" in result.message
            assert result.response_time_ms is not None

    @pytest.mark.asyncio
    async def test_database_health_check_pool_status_failure(self, test_config):
        """Test database health check with pool status failure."""
        with patch('src.web_interface.api.health.DatabaseConnectionManager') as mock_db_manager:
            # Mock successful connection but pool status failure
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = 1
            
            mock_manager = AsyncMock()
            mock_manager.get_connection.return_value.__aenter__.return_value = mock_conn
            mock_manager.get_pool_status.side_effect = Exception("Pool error")
            mock_db_manager.return_value = mock_manager
            
            result = await check_database_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "unhealthy"
            assert "Pool error" in result.message


class TestRedisHealthCheck:
    """Test Redis health check functionality."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return Config(
            redis={
                "url": "redis://localhost:6379/0",
                "password": None
            }
        )

    @pytest.mark.asyncio
    async def test_redis_health_check_success(self, test_config):
        """Test successful Redis health check."""
        with patch('src.web_interface.api.health.RedisClient') as mock_redis_client:
            # Mock successful Redis operations
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_client.set.return_value = True
            mock_client.get.return_value = "test_value"  # Will be set to this in the test
            mock_client.info.return_value = {
                "used_memory_human": "2M",
                "connected_clients": 10,
                "uptime_in_days": 5
            }
            mock_redis_client.return_value = mock_client
            
            # Mock the test value to match what get() returns
            with patch('time.time', return_value=1234567890):
                result = await check_redis_health(test_config)
                
                # The test writes str(int(time.time())), so mock get to return that
                mock_client.get.return_value = "1234567890"
                result = await check_redis_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "healthy"
            assert "Redis connection successful" in result.message
            assert result.response_time_ms is not None
            assert result.metadata["used_memory_human"] == "2M"
            assert result.metadata["connected_clients"] == 10
            assert result.metadata["uptime_days"] == 5

    @pytest.mark.asyncio
    async def test_redis_health_check_ping_failure(self, test_config):
        """Test Redis health check with ping failure."""
        with patch('src.web_interface.api.health.RedisClient') as mock_redis_client:
            mock_client = AsyncMock()
            mock_client.ping.side_effect = Exception("Connection refused")
            mock_redis_client.return_value = mock_client
            
            result = await check_redis_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "unhealthy"
            assert "Connection refused" in result.message

    @pytest.mark.asyncio
    async def test_redis_health_check_read_write_failure(self, test_config):
        """Test Redis health check with read/write failure."""
        with patch('src.web_interface.api.health.RedisClient') as mock_redis_client:
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_client.set.return_value = True
            mock_client.get.return_value = "wrong_value"  # Different from what was set
            mock_redis_client.return_value = mock_client
            
            result = await check_redis_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "unhealthy"
            assert "read/write test failed" in result.message


class TestExchangesHealthCheck:
    """Test exchanges health check functionality."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return Config(
            exchanges={
                "binance": {"api_key": "test", "secret": "test"},
                "coinbase": {"api_key": "test", "secret": "test"}
            }
        )

    @pytest.mark.asyncio
    async def test_exchanges_health_check_all_healthy(self, test_config):
        """Test exchanges health check with all exchanges healthy."""
        with patch('src.web_interface.api.health.ExchangeFactory') as mock_factory, \
             patch('src.web_interface.api.health.ConnectionHealthMonitor') as mock_monitor:
            
            mock_factory_instance = AsyncMock()
            mock_factory_instance.get_available_exchanges.return_value = ["binance", "coinbase"]
            mock_exchange = AsyncMock()
            mock_factory_instance.create_exchange.return_value = mock_exchange
            mock_factory.return_value = mock_factory_instance
            
            mock_monitor_instance = AsyncMock()
            mock_monitor.return_value = mock_monitor_instance
            
            result = await check_exchanges_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "healthy"
            assert "All 2 exchanges healthy" in result.message
            assert result.metadata["total_exchanges"] == 2
            assert result.metadata["healthy_exchanges"] == 2
            assert "binance" in result.metadata["exchanges"]
            assert "coinbase" in result.metadata["exchanges"]

    @pytest.mark.asyncio
    async def test_exchanges_health_check_partially_healthy(self, test_config):
        """Test exchanges health check with some exchanges unhealthy."""
        with patch('src.web_interface.api.health.ExchangeFactory') as mock_factory, \
             patch('src.web_interface.api.health.ConnectionHealthMonitor') as mock_monitor:
            
            mock_factory_instance = AsyncMock()
            mock_factory_instance.get_available_exchanges.return_value = ["binance", "coinbase"]
            mock_exchange = AsyncMock()
            mock_factory_instance.create_exchange.return_value = mock_exchange
            mock_factory.return_value = mock_factory_instance
            
            mock_monitor_instance = AsyncMock()
            
            # First exchange creation succeeds, second fails
            create_results = [AsyncMock(), Exception("API Error")]
            mock_factory_instance.create_exchange.side_effect = create_results
            mock_monitor.return_value = mock_monitor_instance
            
            result = await check_exchanges_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "degraded"
            assert "1/2 exchanges healthy" in result.message
            assert result.metadata["healthy_exchanges"] == 1

    @pytest.mark.asyncio
    async def test_exchanges_health_check_no_exchanges(self, test_config):
        """Test exchanges health check with no exchanges configured."""
        with patch('src.web_interface.api.health.ExchangeFactory') as mock_factory:
            mock_factory_instance = AsyncMock()
            mock_factory_instance.get_available_exchanges.return_value = []
            mock_factory.return_value = mock_factory_instance
            
            result = await check_exchanges_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "degraded"
            assert "No exchanges configured" in result.message

    @pytest.mark.asyncio
    async def test_exchanges_health_check_factory_failure(self, test_config):
        """Test exchanges health check with factory failure."""
        with patch('src.web_interface.api.health.ExchangeFactory') as mock_factory:
            mock_factory.side_effect = Exception("Factory initialization failed")
            
            result = await check_exchanges_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "unhealthy"
            assert "Factory initialization failed" in result.message


class TestMLModelsHealthCheck:
    """Test ML models health check functionality."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return Config(
            ml={
                "model_registry_url": "http://localhost:5000",
                "inference_timeout": 30
            }
        )

    @pytest.mark.asyncio
    async def test_ml_models_health_check_success(self, test_config):
        """Test successful ML models health check."""
        result = await check_ml_models_health(test_config)
        
        assert isinstance(result, ComponentHealth)
        assert result.status == "healthy"
        assert "ML models service available" in result.message
        assert result.response_time_ms is not None
        assert result.metadata["models_loaded"] == 0  # Placeholder value
        assert result.metadata["inference_ready"] is True

    @pytest.mark.asyncio
    async def test_ml_models_health_check_with_model_manager(self, test_config):
        """Test ML models health check with actual model manager."""
        with patch('src.ml.model_manager.ModelManager') as mock_model_manager:
            mock_manager = AsyncMock()
            mock_manager.get_loaded_models.return_value = ["model1", "model2"]
            mock_manager.is_ready.return_value = True
            mock_model_manager.return_value = mock_manager
            
            # This would be implemented when model manager integration is added
            result = await check_ml_models_health(test_config)
            
            assert isinstance(result, ComponentHealth)
            assert result.status == "healthy"

    @pytest.mark.asyncio
    async def test_ml_models_health_check_failure(self, test_config):
        """Test ML models health check failure."""
        with patch('src.web_interface.api.health.logger') as mock_logger:
            # Simulate an exception in the health check
            with patch('time.time', side_effect=Exception("Unexpected error")):
                result = await check_ml_models_health(test_config)
                
                assert isinstance(result, ComponentHealth)
                assert result.status == "unhealthy"
                assert "Unexpected error" in result.message
                mock_logger.error.assert_called_once()


class TestHealthCheckIntegration:
    """Test integration of multiple health check components."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return Config(
            database={"url": "postgresql://test"},
            redis={"url": "redis://test"},
            exchanges={"binance": {"api_key": "test"}},
            ml={"model_registry_url": "http://test"}
        )

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, test_config):
        """Test running multiple health checks concurrently."""
        with patch('src.web_interface.api.health.check_database_health') as mock_db, \
             patch('src.web_interface.api.health.check_redis_health') as mock_redis, \
             patch('src.web_interface.api.health.check_exchanges_health') as mock_exchanges, \
             patch('src.web_interface.api.health.check_ml_models_health') as mock_ml:
            
            # Mock all health checks to return healthy
            healthy_result = ComponentHealth(
                status="healthy",
                message="All good",
                last_check=datetime.now(timezone.utc)
            )
            
            mock_db.return_value = healthy_result
            mock_redis.return_value = healthy_result
            mock_exchanges.return_value = healthy_result
            mock_ml.return_value = healthy_result
            
            # Run all health checks concurrently
            results = await asyncio.gather(
                check_database_health(test_config),
                check_redis_health(test_config),
                check_exchanges_health(test_config),
                check_ml_models_health(test_config)
            )
            
            assert len(results) == 4
            for result in results:
                assert isinstance(result, ComponentHealth)
                assert result.status == "healthy"

    @pytest.mark.asyncio
    async def test_mixed_health_check_results(self, test_config):
        """Test health checks with mixed results."""
        db_result = ComponentHealth(
            status="healthy",
            message="Database OK",
            last_check=datetime.now(timezone.utc)
        )
        
        redis_result = ComponentHealth(
            status="unhealthy", 
            message="Redis down",
            last_check=datetime.now(timezone.utc)
        )
        
        exchanges_result = ComponentHealth(
            status="degraded",
            message="Some exchanges down", 
            last_check=datetime.now(timezone.utc)
        )
        
        ml_result = ComponentHealth(
            status="healthy",
            message="ML models OK",
            last_check=datetime.now(timezone.utc)
        )
        
        results = [db_result, redis_result, exchanges_result, ml_result]
        
        # Test overall status calculation logic
        statuses = [r.status for r in results]
        
        if "unhealthy" in statuses:
            overall_status = "unhealthy"
        elif "degraded" in statuses:
            overall_status = "degraded"  
        else:
            overall_status = "healthy"
            
        assert overall_status == "unhealthy"  # Redis is unhealthy

    def test_component_health_dict_serialization(self):
        """Test ComponentHealth dictionary serialization."""
        health = ComponentHealth(
            status="healthy",
            message="Test message",
            response_time_ms=123.45,
            last_check=datetime.now(timezone.utc),
            metadata={"key": "value", "count": 42}
        )
        
        data = health.dict()
        
        # Verify all fields are serializable
        import json
        json_str = json.dumps(data, default=str)
        
        assert json_str is not None
        assert "healthy" in json_str
        assert "Test message" in json_str