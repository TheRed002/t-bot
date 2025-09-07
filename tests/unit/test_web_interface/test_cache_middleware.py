"""
Test cases for web_interface cache middleware.

This module tests the caching functionality for API responses
in the web interface middleware.
"""

import time
import json
import hashlib
from unittest.mock import Mock, AsyncMock, patch
import pytest
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from src.web_interface.middleware.cache import (
    CacheEntry,
    CacheMiddleware
)


class TestCacheEntry:
    """Test CacheEntry functionality."""

    def test_init(self):
        """Test cache entry initialization."""
        data = {"test": "value"}
        headers = {"Content-Type": "application/json"}
        expires_at = time.time() + 300
        
        entry = CacheEntry(data, headers, expires_at)
        
        assert entry.data == data
        assert entry.headers == headers
        assert entry.expires_at == expires_at
        assert entry.hit_count == 0
        assert entry.created_at <= time.time()
        assert entry.last_access <= time.time()

    def test_is_expired_false(self):
        """Test cache entry not expired."""
        entry = CacheEntry({}, {}, time.time() + 300)
        assert not entry.is_expired()

    def test_is_expired_true(self):
        """Test cache entry is expired."""
        entry = CacheEntry({}, {}, time.time() - 1)
        assert entry.is_expired()

    def test_touch(self):
        """Test touching cache entry."""
        entry = CacheEntry({}, {}, time.time() + 300)
        initial_hit_count = entry.hit_count
        initial_access = entry.last_access
        
        time.sleep(0.01)  # Small delay to ensure time difference
        entry.touch()
        
        assert entry.hit_count == initial_hit_count + 1
        assert entry.last_access > initial_access


class TestCacheMiddleware:
    """Test CacheMiddleware functionality."""

    @pytest.fixture
    def mock_app(self):
        """Create mock FastAPI app."""
        return Mock()

    @pytest.fixture
    def cache_middleware(self, mock_app):
        """Create cache middleware instance."""
        return CacheMiddleware(mock_app, max_cache_size=100, default_ttl=60)

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/test"
        request.url.query = ""
        request.query_params = {}
        request.headers = {}
        request.state = Mock()
        return request

    @pytest.fixture
    def mock_response(self):
        """Create mock response."""
        response = Mock(spec=JSONResponse)
        response.status_code = 200
        response.body = b'{"test": "value"}'
        response.headers = {}
        return response

    def test_init(self, mock_app):
        """Test middleware initialization."""
        middleware = CacheMiddleware(mock_app, max_cache_size=500, default_ttl=120)
        
        assert middleware.max_cache_size == 500
        assert middleware.default_ttl == 120
        assert isinstance(middleware.cache, dict)
        assert len(middleware.cache_config) > 0
        assert len(middleware.no_cache_patterns) > 0

    def test_cache_config_endpoints(self, cache_middleware):
        """Test cache configuration for different endpoints."""
        config = cache_middleware.cache_config
        
        # Check specific configurations
        assert config["/api/trading/market-data"]["ttl"] == 5
        assert config["/api/portfolio/summary"]["ttl"] == 30
        assert config["/api/strategies"]["ttl"] == 300
        assert config["/api/ml/models"]["ttl"] == 600

    def test_no_cache_patterns(self, cache_middleware):
        """Test no-cache patterns."""
        patterns = cache_middleware.no_cache_patterns
        
        assert "/api/trading/orders" in patterns
        assert "/auth/" in patterns
        assert "/ws/" in patterns

    async def test_dispatch_bypass_cache_post(self, cache_middleware, mock_request):
        """Test dispatch bypasses cache for POST requests."""
        mock_request.method = "POST"
        mock_call_next = AsyncMock(return_value=Mock())
        
        with patch.object(cache_middleware, '_should_bypass_cache', return_value=True):
            response = await cache_middleware.dispatch(mock_request, mock_call_next)
            
        mock_call_next.assert_called_once()

    async def test_dispatch_cache_hit(self, cache_middleware, mock_request):
        """Test dispatch returns cached response."""
        cached_response = JSONResponse({"cached": True})
        
        with patch.object(cache_middleware, '_should_bypass_cache', return_value=False), \
             patch.object(cache_middleware, '_generate_cache_key', return_value="test_key"), \
             patch.object(cache_middleware, '_get_cached_response', return_value=cached_response), \
             patch.object(cache_middleware, '_add_cache_headers'):
            
            response = await cache_middleware.dispatch(mock_request, AsyncMock())
            assert response == cached_response

    async def test_dispatch_cache_miss_and_store(self, cache_middleware, mock_request, mock_response):
        """Test dispatch handles cache miss and stores response."""
        mock_call_next = AsyncMock(return_value=mock_response)
        
        with patch.object(cache_middleware, '_should_bypass_cache', return_value=False), \
             patch.object(cache_middleware, '_generate_cache_key', return_value="test_key"), \
             patch.object(cache_middleware, '_get_cached_response', return_value=None), \
             patch.object(cache_middleware, '_should_cache_response', return_value=True), \
             patch.object(cache_middleware, '_cache_response') as mock_cache, \
             patch.object(cache_middleware, '_add_cache_headers'):
            
            response = await cache_middleware.dispatch(mock_request, mock_call_next)
            
            mock_cache.assert_called_once()
            assert response == mock_response

    def test_should_bypass_cache_non_get(self, cache_middleware):
        """Test bypass cache for non-GET requests."""
        request = Mock()
        request.method = "POST"
        
        assert cache_middleware._should_bypass_cache(request)

    def test_should_bypass_cache_no_cache_pattern(self, cache_middleware):
        """Test bypass cache for no-cache patterns."""
        request = Mock()
        request.method = "GET"
        request.url.path = "/api/trading/orders"
        request.headers = {}
        request.query_params = {}
        
        assert cache_middleware._should_bypass_cache(request)

    def test_should_bypass_cache_no_cache_header(self, cache_middleware):
        """Test bypass cache with no-cache header."""
        request = Mock()
        request.method = "GET"
        request.url.path = "/api/test"
        request.headers = {"Cache-Control": "no-cache"}
        request.query_params = {}
        
        assert cache_middleware._should_bypass_cache(request)

    def test_should_bypass_cache_realtime_param(self, cache_middleware):
        """Test bypass cache with real-time parameter."""
        request = Mock()
        request.method = "GET"
        request.url.path = "/api/test"
        request.headers = {}
        request.query_params = {"real-time": "true"}
        
        assert cache_middleware._should_bypass_cache(request)

    def test_should_bypass_cache_normal_request(self, cache_middleware):
        """Test normal request doesn't bypass cache."""
        request = Mock()
        request.method = "GET"
        request.url.path = "/api/portfolio/summary"
        request.headers = {}
        request.query_params = {}
        
        assert not cache_middleware._should_bypass_cache(request)

    def test_should_cache_response_error(self, cache_middleware, mock_request):
        """Test should not cache error responses."""
        response = Mock()
        response.status_code = 404
        
        assert not cache_middleware._should_cache_response(mock_request, response)

    def test_should_cache_response_disabled(self, cache_middleware, mock_request):
        """Test should not cache when disabled for endpoint."""
        response = Mock()
        response.status_code = 200
        mock_request.url.path = "/api/unknown"
        
        assert not cache_middleware._should_cache_response(mock_request, response)

    def test_should_cache_response_enabled(self, cache_middleware, mock_request):
        """Test should cache when enabled for endpoint."""
        response = Mock()
        response.status_code = 200
        mock_request.url.path = "/api/portfolio/summary"
        
        assert cache_middleware._should_cache_response(mock_request, response)

    def test_get_endpoint_config_exact_match(self, cache_middleware):
        """Test get endpoint config with exact match."""
        config = cache_middleware._get_endpoint_config("/api/portfolio/summary")
        
        assert config["ttl"] == 30
        assert config["enabled"]

    def test_get_endpoint_config_pattern_match(self, cache_middleware):
        """Test get endpoint config with pattern matching."""
        # Add a pattern for testing
        cache_middleware.cache_config["/api/test/*"] = {"ttl": 120, "enabled": True}
        
        config = cache_middleware._get_endpoint_config("/api/test/specific")
        assert config["ttl"] == 120

    def test_get_endpoint_config_default(self, cache_middleware):
        """Test get endpoint config returns default."""
        config = cache_middleware._get_endpoint_config("/api/unknown")
        
        assert config["ttl"] == 60  # default_ttl
        assert not config["enabled"]

    def test_generate_cache_key_basic(self, cache_middleware):
        """Test generate cache key for basic request."""
        request = Mock()
        request.method = "GET"
        request.url.path = "/api/test"
        request.query_params = {}
        request.state = Mock()
        
        # No user state - should not include user ID
        key = cache_middleware._generate_cache_key(request)
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length

    def test_generate_cache_key_with_params(self, cache_middleware):
        """Test generate cache key with query parameters."""
        request = Mock()
        request.method = "GET"
        request.url.path = "/api/test"
        request.query_params = {"param1": "value1", "param2": "value2"}
        request.state = Mock()
        
        key = cache_middleware._generate_cache_key(request)
        assert isinstance(key, str)

    def test_generate_cache_key_with_user(self, cache_middleware):
        """Test generate cache key with user context."""
        request = Mock()
        request.method = "GET"
        request.url.path = "/api/test"
        request.query_params = {}
        request.state = Mock()
        request.state.user = {"user_id": "123"}
        
        key_with_user = cache_middleware._generate_cache_key(request)
        assert isinstance(key_with_user, str)
        assert len(key_with_user) == 32  # MD5 hash length
        
        # Generate key without user and verify they're different
        request_no_user = Mock()
        request_no_user.method = "GET"
        request_no_user.url.path = "/api/test"
        request_no_user.query_params = {}
        request_no_user.state = Mock()
        
        key_without_user = cache_middleware._generate_cache_key(request_no_user)
        assert key_with_user != key_without_user  # Different keys due to user context

    def test_get_cached_response_not_found(self, cache_middleware):
        """Test get cached response when not found."""
        response = cache_middleware._get_cached_response("nonexistent_key")
        assert response is None

    def test_get_cached_response_expired(self, cache_middleware):
        """Test get cached response when expired."""
        expired_entry = CacheEntry({}, {}, time.time() - 1)
        cache_middleware.cache["test_key"] = expired_entry
        
        response = cache_middleware._get_cached_response("test_key")
        
        assert response is None
        assert "test_key" not in cache_middleware.cache

    def test_get_cached_response_valid(self, cache_middleware):
        """Test get cached response when valid."""
        data = {"test": "value"}
        headers = {"Content-Type": "application/json"}
        valid_entry = CacheEntry(data, headers, time.time() + 300)
        cache_middleware.cache["test_key"] = valid_entry
        
        response = cache_middleware._get_cached_response("test_key")
        
        assert isinstance(response, JSONResponse)
        assert valid_entry.hit_count == 1

    async def test_cache_response_json_response(self, cache_middleware, mock_request):
        """Test cache response with JSON response."""
        response = JSONResponse({"test": "value"})
        response.body = json.dumps({"test": "value"}).encode()
        
        await cache_middleware._cache_response("test_key", mock_request, response)
        
        assert "test_key" in cache_middleware.cache
        entry = cache_middleware.cache["test_key"]
        assert entry.data == {"test": "value"}

    async def test_cache_response_non_json(self, cache_middleware, mock_request):
        """Test cache response with non-JSON response."""
        response = Mock()
        response.body = b"not json"
        
        await cache_middleware._cache_response("test_key", mock_request, response)
        
        # Should not cache non-JSON responses
        assert "test_key" not in cache_middleware.cache

    async def test_cache_response_eviction(self, cache_middleware, mock_request):
        """Test cache response triggers eviction when full."""
        # Fill cache to capacity
        for i in range(100):
            cache_middleware.cache[f"key_{i}"] = CacheEntry({}, {}, time.time() + 300)
        
        response = JSONResponse({"test": "value"})
        response.body = json.dumps({"test": "value"}).encode()
        
        with patch.object(cache_middleware, '_evict_lru_entries') as mock_evict:
            await cache_middleware._cache_response("new_key", mock_request, response)
            mock_evict.assert_called_once()

    def test_evict_lru_entries_empty_cache(self, cache_middleware):
        """Test evict LRU entries with empty cache."""
        cache_middleware._evict_lru_entries()
        # Should not raise exception

    def test_evict_lru_entries_normal(self, cache_middleware):
        """Test evict LRU entries removes oldest."""
        # Add entries with different access times
        old_entry = CacheEntry({}, {}, time.time() + 300)
        old_entry.last_access = time.time() - 100
        
        new_entry = CacheEntry({}, {}, time.time() + 300)
        new_entry.last_access = time.time()
        
        cache_middleware.cache["old_key"] = old_entry
        cache_middleware.cache["new_key"] = new_entry
        
        cache_middleware._evict_lru_entries()
        
        # Old entry should be evicted
        assert "old_key" not in cache_middleware.cache
        assert "new_key" in cache_middleware.cache

    def test_handle_cache_invalidation_exact_match(self, cache_middleware):
        """Test cache invalidation with exact match."""
        request = Mock()
        request.method = "POST"
        request.url.path = "/api/trading/orders"
        
        with patch.object(cache_middleware, '_invalidate_cache_patterns') as mock_invalidate:
            cache_middleware._handle_cache_invalidation(request)
            mock_invalidate.assert_called_once()

    def test_handle_cache_invalidation_pattern_match(self, cache_middleware):
        """Test cache invalidation with pattern matching."""
        request = Mock()
        request.method = "POST"
        request.url.path = "/api/bots/start"
        
        with patch.object(cache_middleware, '_matches_trigger_pattern', return_value=True), \
             patch.object(cache_middleware, '_invalidate_cache_patterns') as mock_invalidate:
            cache_middleware._handle_cache_invalidation(request)
            mock_invalidate.assert_called_once()

    def test_matches_trigger_pattern_wildcard(self, cache_middleware):
        """Test matches trigger pattern with wildcard."""
        assert cache_middleware._matches_trigger_pattern("POST /api/test/123", "POST /api/test/*")
        assert not cache_middleware._matches_trigger_pattern("GET /api/test/123", "POST /api/test/*")

    def test_matches_trigger_pattern_exact(self, cache_middleware):
        """Test matches trigger pattern exact match."""
        assert cache_middleware._matches_trigger_pattern("POST /api/test", "POST /api/test")
        assert not cache_middleware._matches_trigger_pattern("POST /api/other", "POST /api/test")

    def test_invalidate_cache_patterns(self, cache_middleware):
        """Test invalidate cache patterns."""
        # Add some cache entries with paths that would match patterns
        cache_middleware.cache["portfolio_key"] = CacheEntry({}, {}, time.time() + 300, "/api/portfolio/summary")
        cache_middleware.cache["other_key"] = CacheEntry({}, {}, time.time() + 300, "/api/other/data")
        
        cache_middleware._invalidate_cache_patterns(["/api/portfolio/"])
        
        # Should remove portfolio-related entries
        assert "portfolio_key" not in cache_middleware.cache
        assert "other_key" in cache_middleware.cache

    def test_add_cache_headers_hit(self, cache_middleware):
        """Test add cache headers for cache hit."""
        response = Mock()
        response.headers = {}
        
        cache_middleware._add_cache_headers(response, hit=True)
        
        assert response.headers["X-Cache"] == "HIT"
        assert "X-Cache-Time" in response.headers

    def test_add_cache_headers_miss(self, cache_middleware):
        """Test add cache headers for cache miss."""
        response = Mock()
        response.headers = {}
        response.body = b'{"test": "value"}'
        
        cache_middleware._add_cache_headers(response, hit=False)
        
        assert response.headers["X-Cache"] == "MISS"
        assert "Cache-Control" in response.headers
        assert "ETag" in response.headers

    def test_get_cache_stats_empty(self, cache_middleware):
        """Test get cache stats with empty cache."""
        stats = cache_middleware.get_cache_stats()
        
        assert stats["total_entries"] == 0
        assert stats["expired_entries"] == 0
        assert stats["total_hits"] == 0
        assert stats["average_hits_per_entry"] == 0

    def test_get_cache_stats_with_entries(self, cache_middleware):
        """Test get cache stats with cache entries."""
        entry1 = CacheEntry({}, {}, time.time() + 300)
        entry1.hit_count = 5
        entry2 = CacheEntry({}, {}, time.time() - 1)  # expired
        entry2.hit_count = 3
        
        cache_middleware.cache["key1"] = entry1
        cache_middleware.cache["key2"] = entry2
        
        stats = cache_middleware.get_cache_stats()
        
        assert stats["total_entries"] == 2
        assert stats["expired_entries"] == 1
        assert stats["active_entries"] == 1
        assert stats["total_hits"] == 8
        assert stats["average_hits_per_entry"] == 4.0

    def test_clear_cache_all(self, cache_middleware):
        """Test clear all cache entries."""
        cache_middleware.cache["key1"] = CacheEntry({}, {}, time.time() + 300)
        cache_middleware.cache["key2"] = CacheEntry({}, {}, time.time() + 300)
        
        cache_middleware.clear_cache()
        
        assert len(cache_middleware.cache) == 0

    def test_clear_cache_pattern(self, cache_middleware):
        """Test clear cache entries by pattern."""
        cache_middleware.cache["portfolio_key"] = CacheEntry({}, {}, time.time() + 300)
        cache_middleware.cache["trading_key"] = CacheEntry({}, {}, time.time() + 300)
        
        cache_middleware.clear_cache("portfolio")
        
        assert "portfolio_key" not in cache_middleware.cache
        assert "trading_key" in cache_middleware.cache