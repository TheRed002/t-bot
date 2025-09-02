"""
Unit tests for ML model cache service.
"""

from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

import pytest

from src.ml.inference.model_cache import (
    ModelCacheConfig,
    ModelCacheService,
)


@pytest.fixture
def sample_config():
    """Sample configuration for tests."""
    return {
        "model_cache": {
            "model_cache_size": 5,
            "max_memory_gb": 1.0,
            "prediction_cache_ttl_minutes": 30,
            "enable_memory_monitoring": True,
            "cleanup_interval_seconds": 60,
            "memory_pressure_threshold": 80.0,
        }
    }


class TestModelCacheConfig:
    """Test model cache configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelCacheConfig()
        
        assert config.model_cache_size == 10
        assert config.max_memory_gb == 2.0
        assert config.prediction_cache_ttl_minutes == 60
        assert config.enable_memory_monitoring is True
        assert config.cleanup_interval_seconds == 30
        assert config.memory_pressure_threshold == 85.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ModelCacheConfig(
            model_cache_size=20,
            max_memory_gb=4.0,
            enable_memory_monitoring=False,
        )
        
        assert config.model_cache_size == 20
        assert config.max_memory_gb == 4.0
        assert config.enable_memory_monitoring is False


class TestModelCacheService:
    """Test model cache service."""

    @pytest.fixture
    def service(self, sample_config):
        """Create service instance for tests."""
        return ModelCacheService(config=sample_config)

    def test_initialization(self, service):
        """Test service initialization."""
        assert service.name == "ModelCacheService"
        assert isinstance(service.mc_config, ModelCacheConfig)
        assert service.mc_config.model_cache_size == 5

    def test_initialization_no_config(self):
        """Test initialization without config."""
        service = ModelCacheService()
        assert isinstance(service.mc_config, ModelCacheConfig)
        # Should use defaults
        assert service.mc_config.model_cache_size == 10

    @pytest.mark.asyncio
    async def test_start_service(self, service):
        """Test service startup."""
        await service.start()
        
        assert service.is_running
        
        # Cleanup
        await service.stop()

    @pytest.mark.asyncio 
    async def test_stop_service(self, service):
        """Test service shutdown."""
        await service.start()
        await service.stop()
        
        assert not service.is_running

    def test_get_cache_size(self, service):
        """Test getting cache size."""
        size = service.get_cache_size()
        assert isinstance(size, int)
        assert size >= 0

    def test_get_cache_memory_usage(self, service):
        """Test getting cache memory usage."""
        memory = service.get_cache_memory_usage()
        assert isinstance(memory, (int, float))
        assert memory >= 0

    def test_clear_cache(self, service):
        """Test clearing cache."""
        # Should not raise exception
        service.clear_cache()

    def test_get_cache_statistics(self, service):
        """Test getting cache statistics."""
        stats = service.get_cache_statistics()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_cache_model_basic(self, service):
        """Test basic model caching."""
        model_id = "test_model"
        mock_model = Mock()
        
        # Should not raise exception
        await service.cache_model(model_id, mock_model)

    @pytest.mark.asyncio
    async def test_get_model_basic(self, service):
        """Test basic model retrieval."""
        model_id = "test_model"
        
        # Should return None for non-existent model
        result = await service.get_model(model_id)
        # Result could be None or some default behavior

    @pytest.mark.asyncio
    async def test_evict_model_basic(self, service):
        """Test basic model eviction."""
        model_id = "test_model"
        
        # Should not raise exception
        result = await service.evict_model(model_id)
        assert isinstance(result, bool)

    def test_is_model_cached(self, service):
        """Test checking if model is cached."""
        model_id = "test_model"
        
        result = service.is_model_cached(model_id)
        assert isinstance(result, bool)

    def test_get_cached_model_ids(self, service):
        """Test getting cached model IDs."""
        result = service.get_cached_model_ids()
        assert isinstance(result, list)


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def service(self, sample_config):
        """Create service for error testing."""
        return ModelCacheService(config=sample_config)

    @pytest.mark.asyncio
    async def test_cache_none_model(self, service):
        """Test caching None model."""
        # Should handle gracefully
        await service.cache_model("test", None)

    @pytest.mark.asyncio
    async def test_get_nonexistent_model(self, service):
        """Test getting non-existent model."""
        result = await service.get_model("nonexistent")
        # Should not raise exception

    def test_clear_empty_cache(self, service):
        """Test clearing empty cache."""
        # Should not raise exception
        service.clear_cache()

    @pytest.mark.asyncio
    async def test_evict_nonexistent_model(self, service):
        """Test evicting non-existent model."""
        result = await service.evict_model("nonexistent")
        assert isinstance(result, bool)