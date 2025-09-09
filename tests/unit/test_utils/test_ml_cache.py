"""
Tests for ML Cache utilities.
"""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.utils.ml_cache import (
    TTLCache,
    ModelCache,
    PredictionCache,
    FeatureCache,
    CacheManager,
    generate_cache_key,
    generate_model_cache_key,
    generate_prediction_cache_key,
    generate_feature_cache_key,
    get_cache_manager,
    init_cache_manager,
    _cache_manager
)


class TestTTLCache:
    """Test TTLCache functionality."""

    @pytest.mark.asyncio
    async def test_ttl_cache_initialization(self):
        """Test TTL cache initialization."""
        cache = TTLCache[str](ttl_seconds=300, max_size=100)
        assert cache.ttl_seconds == 300
        assert cache.max_size == 100
        assert cache._cache == {}
        assert cache._lock is not None

    @pytest.mark.asyncio
    async def test_get_set_operations(self):
        """Test basic get/set operations."""
        cache = TTLCache[str](ttl_seconds=300)
        
        # Test get on empty cache
        result = await cache.get("key1")
        assert result is None
        
        # Test set and get
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = TTLCache[str](ttl_seconds=1)  # 1 second TTL
        
        await cache.set("key1", "value1")
        
        # Should be available immediately
        result = await cache.get("key1")
        assert result == "value1"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_max_size_limit(self):
        """Test max size limit enforcement."""
        cache = TTLCache[str](ttl_seconds=300, max_size=2)
        
        # Add up to max size
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        # Both should be present
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"
        
        # Add one more to trigger eviction
        await cache.set("key3", "value3")
        
        # Oldest should be evicted
        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_delete_operation(self):
        """Test delete operation."""
        cache = TTLCache[str](ttl_seconds=300)
        
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"
        
        # Test successful deletion
        result = await cache.delete("key1")
        assert result is True
        assert await cache.get("key1") is None
        
        # Test deletion of non-existent key
        result = await cache.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_operation(self):
        """Test clear operation."""
        cache = TTLCache[str](ttl_seconds=300)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        count = await cache.clear()
        assert count == 2
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_size_operation(self):
        """Test size operation."""
        cache = TTLCache[str](ttl_seconds=300)
        
        assert await cache.size() == 0
        
        await cache.set("key1", "value1")
        assert await cache.size() == 1
        
        await cache.set("key2", "value2")
        assert await cache.size() == 2

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleanup expired entries."""
        cache = TTLCache[str](ttl_seconds=1)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Add one more that's not expired
        await cache.set("key3", "value3")
        
        # Cleanup should remove expired entries
        count = await cache.cleanup_expired()
        assert count == 2
        assert await cache.size() == 1
        assert await cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_set_update_existing(self):
        """Test updating existing key doesn't count toward size limit."""
        cache = TTLCache[str](ttl_seconds=300, max_size=2)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        # Update existing key - should not trigger eviction
        await cache.set("key1", "value1_updated")
        
        assert await cache.get("key1") == "value1_updated"
        assert await cache.get("key2") == "value2"
        assert await cache.size() == 2


class TestModelCache:
    """Test ModelCache functionality."""

    @pytest.mark.asyncio
    async def test_model_cache_initialization(self):
        """Test model cache initialization."""
        cache = ModelCache(ttl_hours=12, max_models=50)
        assert cache.ttl_seconds == 12 * 3600
        assert cache.max_size == 50

    @pytest.mark.asyncio
    async def test_model_cache_operations(self):
        """Test model cache operations."""
        cache = ModelCache()
        model = {"type": "test_model", "params": [1, 2, 3]}
        
        # Test get on empty cache
        result = await cache.get_model("model1")
        assert result is None
        
        # Test cache and retrieve model
        await cache.cache_model("model1", model)
        result = await cache.get_model("model1")
        assert result == model
        
        # Test remove model
        removed = await cache.remove_model("model1")
        assert removed is True
        assert await cache.get_model("model1") is None
        
        # Test remove non-existent model
        removed = await cache.remove_model("nonexistent")
        assert removed is False


class TestPredictionCache:
    """Test PredictionCache functionality."""

    @pytest.mark.asyncio
    async def test_prediction_cache_initialization(self):
        """Test prediction cache initialization."""
        cache = PredictionCache(ttl_minutes=10, max_predictions=5000)
        assert cache.ttl_seconds == 10 * 60
        assert cache.max_size == 5000

    @pytest.mark.asyncio
    async def test_prediction_cache_operations(self):
        """Test prediction cache operations."""
        cache = PredictionCache()
        prediction = {"prediction": 0.85, "confidence": 0.92}
        
        # Test get on empty cache
        result = await cache.get_prediction("pred1")
        assert result is None
        
        # Test cache and retrieve prediction
        await cache.cache_prediction("pred1", prediction)
        result = await cache.get_prediction("pred1")
        assert result == prediction


class TestFeatureCache:
    """Test FeatureCache functionality."""

    @pytest.mark.asyncio
    async def test_feature_cache_initialization(self):
        """Test feature cache initialization."""
        cache = FeatureCache(ttl_hours=6, max_feature_sets=500)
        assert cache.ttl_seconds == 6 * 3600
        assert cache.max_size == 500

    @pytest.mark.asyncio
    async def test_feature_cache_operations(self):
        """Test feature cache operations."""
        cache = FeatureCache()
        features = {"feature1": 1.0, "feature2": 2.0}
        
        # Test get on empty cache
        result = await cache.get_features("feat1")
        assert result is None
        
        # Test cache and retrieve features
        await cache.cache_features("feat1", features)
        result = await cache.get_features("feat1")
        assert result == features


class TestCacheKeyGeneration:
    """Test cache key generation functions."""

    def test_generate_cache_key_simple(self):
        """Test simple cache key generation."""
        key = generate_cache_key("arg1", "arg2", 123)
        assert isinstance(key, str)
        assert len(key) == 16  # MD5 hash truncated to 16 chars
        
        # Same arguments should produce same key
        key2 = generate_cache_key("arg1", "arg2", 123)
        assert key == key2

    def test_generate_cache_key_with_dict(self):
        """Test cache key generation with dict."""
        key = generate_cache_key({"a": 1, "b": 2})
        assert isinstance(key, str)
        assert len(key) == 16

    def test_generate_cache_key_with_list(self):
        """Test cache key generation with list."""
        key = generate_cache_key([1, 2, 3])
        assert isinstance(key, str)
        assert len(key) == 16

    def test_generate_cache_key_with_pandas_like_object(self):
        """Test cache key generation with pandas-like object."""
        # Mock pandas-like object
        mock_obj = MagicMock()
        mock_obj.shape = (10, 5)
        mock_obj.iloc = MagicMock()
        mock_obj.iloc.__len__ = MagicMock(return_value=1)
        mock_obj.iloc.__getitem__ = MagicMock(return_value=MagicMock(values=[1, 2, 3]))
        
        key = generate_cache_key(mock_obj)
        assert isinstance(key, str)
        assert len(key) == 16

    def test_generate_model_cache_key(self):
        """Test model cache key generation."""
        key = generate_model_cache_key("model123")
        assert key == "model_model123"
        
        key_with_version = generate_model_cache_key("model123", "v1.0")
        assert key_with_version == "model_model123_v1.0"

    def test_generate_prediction_cache_key(self):
        """Test prediction cache key generation."""
        key = generate_prediction_cache_key("model123", "hash456", False)
        assert key == "pred_model123_hash456_False"
        
        key_with_probs = generate_prediction_cache_key("model123", "hash456", True)
        assert key_with_probs == "pred_model123_hash456_True"

    def test_generate_feature_cache_key(self):
        """Test feature cache key generation."""
        key = generate_feature_cache_key("BTCUSD", ["technical", "statistical"], "hash789")
        assert key == "feat_BTCUSD_statistical_technical_hash789"


class TestCacheManager:
    """Test CacheManager functionality."""

    def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        config = {
            "model_cache_ttl_hours": 12,
            "max_cached_models": 50,
            "prediction_cache_ttl_minutes": 10,
            "max_cached_predictions": 5000,
            "feature_cache_ttl_hours": 6,
            "max_cached_feature_sets": 500,
            "cleanup_interval_minutes": 15
        }
        
        manager = CacheManager(config)
        assert manager.model_cache.ttl_seconds == 12 * 3600
        assert manager.model_cache.max_size == 50
        assert manager.prediction_cache.ttl_seconds == 10 * 60
        assert manager.prediction_cache.max_size == 5000
        assert manager.feature_cache.ttl_seconds == 6 * 3600
        assert manager.feature_cache.max_size == 500
        assert manager._cleanup_interval == 15

    def test_cache_manager_default_config(self):
        """Test cache manager with default config."""
        manager = CacheManager()
        assert manager.model_cache.ttl_seconds == 24 * 3600
        assert manager.model_cache.max_size == 100

    @pytest.mark.asyncio
    async def test_cache_manager_start_stop(self):
        """Test cache manager start/stop."""
        manager = CacheManager({"cleanup_interval_minutes": 1})
        
        await manager.start()
        assert manager._cleanup_task is not None
        assert not manager._cleanup_task.cancelled()
        
        await manager.stop()
        assert manager._cleanup_task.cancelled()

    @pytest.mark.asyncio
    async def test_cache_manager_start_no_cleanup(self):
        """Test cache manager start with cleanup disabled."""
        manager = CacheManager({"cleanup_interval_minutes": 0})
        
        await manager.start()
        assert manager._cleanup_task is None
        
        await manager.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_clear_all(self):
        """Test clearing all caches."""
        manager = CacheManager()
        
        # Add some data
        await manager.model_cache.cache_model("model1", {"test": "model"})
        await manager.prediction_cache.cache_prediction("pred1", {"test": "pred"})
        await manager.feature_cache.cache_features("feat1", {"test": "feat"})
        
        results = await manager.clear_all()
        
        assert results["models_cleared"] == 1
        assert results["predictions_cleared"] == 1
        assert results["features_cleared"] == 1

    @pytest.mark.asyncio
    async def test_get_cache_stats(self):
        """Test getting cache statistics."""
        manager = CacheManager()
        
        # Add some data
        await manager.model_cache.cache_model("model1", {"test": "model"})
        await manager.prediction_cache.cache_prediction("pred1", {"test": "pred"})
        
        stats = await manager.get_cache_stats()
        
        assert stats["model_cache_size"] == 1
        assert stats["prediction_cache_size"] == 1
        assert stats["feature_cache_size"] == 0
        assert stats["model_cache_ttl_hours"] == 24
        assert stats["prediction_cache_ttl_minutes"] == 5
        assert stats["feature_cache_ttl_hours"] == 4

    @pytest.mark.asyncio
    async def test_background_cleanup(self):
        """Test background cleanup task."""
        manager = CacheManager({"cleanup_interval_minutes": 0.001})  # 0.06 seconds
        
        # Mock the cleanup methods
        manager.model_cache.cleanup_expired = AsyncMock(return_value=1)
        manager.prediction_cache.cleanup_expired = AsyncMock(return_value=2)
        manager.feature_cache.cleanup_expired = AsyncMock(return_value=0)
        
        await manager.start()
        
        # Wait for at least one cleanup cycle
        await asyncio.sleep(0.15)
        
        await manager.stop()
        
        # Verify cleanup was called
        manager.model_cache.cleanup_expired.assert_called()
        manager.prediction_cache.cleanup_expired.assert_called()
        manager.feature_cache.cleanup_expired.assert_called()

    @pytest.mark.asyncio
    async def test_background_cleanup_exception_handling(self):
        """Test background cleanup exception handling."""
        manager = CacheManager({"cleanup_interval_minutes": 0.001})
        
        # Mock cleanup to raise exception
        manager.model_cache.cleanup_expired = AsyncMock(side_effect=Exception("Test error"))
        manager.prediction_cache.cleanup_expired = AsyncMock(return_value=0)
        manager.feature_cache.cleanup_expired = AsyncMock(return_value=0)
        
        with patch('src.utils.ml_cache.logger') as mock_logger:
            await manager.start()
            await asyncio.sleep(0.15)
            await manager.stop()
            
            # Verify error was logged
            mock_logger.error.assert_called()


class TestGlobalCacheManager:
    """Test global cache manager functions."""

    def test_get_cache_manager_singleton(self):
        """Test cache manager singleton behavior."""
        # Clear global instance
        import src.utils.ml_cache
        src.utils.ml_cache._cache_manager = None
        
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, CacheManager)

    @pytest.mark.asyncio
    async def test_init_cache_manager(self):
        """Test cache manager initialization."""
        # Clear global instance
        import src.utils.ml_cache
        src.utils.ml_cache._cache_manager = None
        
        config = {"model_cache_ttl_hours": 12}
        
        with patch.object(CacheManager, 'start') as mock_start:
            manager = await init_cache_manager(config)
            
            assert isinstance(manager, CacheManager)
            assert manager.model_cache.ttl_seconds == 12 * 3600
            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_cache_manager_no_config(self):
        """Test cache manager initialization without config."""
        # Clear global instance
        import src.utils.ml_cache
        src.utils.ml_cache._cache_manager = None
        
        with patch.object(CacheManager, 'start') as mock_start:
            manager = await init_cache_manager()
            
            assert isinstance(manager, CacheManager)
            mock_start.assert_called_once()


class TestCacheKeyEdgeCases:
    """Test edge cases in cache key generation."""

    def test_generate_cache_key_empty_args(self):
        """Test cache key generation with empty arguments."""
        key = generate_cache_key()
        assert isinstance(key, str)
        assert len(key) == 16

    def test_generate_cache_key_none_values(self):
        """Test cache key generation with None values."""
        key = generate_cache_key(None, "test", None)
        assert isinstance(key, str)
        assert len(key) == 16

    def test_generate_cache_key_complex_objects(self):
        """Test cache key generation with complex nested objects."""
        key = generate_cache_key(
            {"nested": {"dict": [1, 2, 3]}},
            [{"list": "item"}],
            "string"
        )
        assert isinstance(key, str)
        assert len(key) == 16

    def test_generate_cache_key_pandas_empty_dataframe(self):
        """Test cache key generation with empty pandas-like object."""
        mock_obj = MagicMock()
        mock_obj.shape = (0, 5)
        mock_obj.iloc = MagicMock()
        mock_obj.iloc.__len__ = MagicMock(return_value=0)
        
        key = generate_cache_key(mock_obj)
        assert isinstance(key, str)
        assert len(key) == 16

    def test_generate_cache_key_pandas_no_values_attr(self):
        """Test cache key generation with pandas-like object without values attribute."""
        mock_obj = MagicMock()
        mock_obj.shape = (1, 5)
        mock_obj.iloc = MagicMock()
        mock_obj.iloc.__len__ = MagicMock(return_value=1)
        mock_iloc_item = MagicMock()
        del mock_iloc_item.values  # Remove values attribute
        mock_obj.iloc.__getitem__ = MagicMock(return_value=mock_iloc_item)
        
        key = generate_cache_key(mock_obj)
        assert isinstance(key, str)
        assert len(key) == 16