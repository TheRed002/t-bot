"""
Caching Middleware for T-Bot web interface.

This middleware provides intelligent caching for API responses to improve
performance while ensuring data freshness for trading operations.
"""

import hashlib
import json
import time
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import get_logger

logger = get_logger(__name__)


class CacheEntry:
    """Represents a cached response entry."""

    def __init__(self, data: Any, headers: dict[str, str], expires_at: float, path: str = ""):
        self.data = data
        self.headers = headers
        self.expires_at = expires_at
        self.path = path
        self.created_at = time.time()
        self.hit_count = 0
        self.last_access = time.time()

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() > self.expires_at

    def touch(self):
        """Update last access time and increment hit count."""
        self.last_access = time.time()
        self.hit_count += 1


class CacheMiddleware(BaseHTTPMiddleware):
    """
    Intelligent caching middleware for API responses.

    This middleware provides:
    - Configurable TTL for different endpoint types
    - Cache invalidation based on trading operations
    - Memory-efficient cache with LRU eviction
    - Cache headers for client-side caching
    - Bypass mechanisms for real-time data
    """

    def __init__(self, app, max_cache_size: int = 1000, default_ttl: int = 60):
        """
        Initialize caching middleware.

        Args:
            app: FastAPI application
            max_cache_size: Maximum number of cached entries
            default_ttl: Default TTL in seconds
        """
        super().__init__(app)
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        self.cache: dict[str, CacheEntry] = {}

        # Endpoint-specific cache configurations
        self.cache_config = {
            # Market data - short TTL due to volatility
            "/api/trading/market-data": {"ttl": 5, "enabled": True},
            "/api/trading/orderbook": {"ttl": 2, "enabled": True},
            # Portfolio data - medium TTL
            "/api/portfolio/summary": {"ttl": 30, "enabled": True},
            "/api/portfolio/positions": {"ttl": 15, "enabled": True},
            "/api/portfolio/balances": {"ttl": 30, "enabled": True},
            "/api/portfolio/pnl": {"ttl": 60, "enabled": True},
            # Risk metrics - medium TTL
            "/api/risk/metrics": {"ttl": 60, "enabled": True},
            "/api/risk/limits": {"ttl": 300, "enabled": True},  # 5 minutes
            "/api/risk/correlation-matrix": {"ttl": 600, "enabled": True},  # 10 minutes
            # Strategy data - longer TTL
            "/api/strategies": {"ttl": 300, "enabled": True},
            "/api/strategies/performance": {"ttl": 300, "enabled": True},
            # ML models - longer TTL
            "/api/ml/models": {"ttl": 600, "enabled": True},
            "/api/ml/performance": {"ttl": 300, "enabled": True},
            # Monitoring data - short TTL
            "/api/monitoring/health": {"ttl": 10, "enabled": True},
            "/api/monitoring/metrics": {"ttl": 30, "enabled": True},
            "/api/monitoring/performance": {"ttl": 60, "enabled": True},
            # Bot management - medium TTL
            "/api/bots": {"ttl": 30, "enabled": True},
            "/api/bots/orchestrator/status": {"ttl": 15, "enabled": True},
        }

        # Endpoints that should never be cached
        self.no_cache_patterns = {
            "/api/trading/orders",  # Order operations
            "/api/bots/start",  # Bot actions
            "/api/bots/stop",  # Bot actions
            "/auth/",  # Authentication
            "/ws/",  # WebSocket endpoints
        }

        # Cache invalidation triggers
        self.invalidation_triggers = {
            # Order operations invalidate trading and portfolio data
            "POST /api/trading/orders": [
                "/api/portfolio/",
                "/api/trading/orders",
                "/api/risk/metrics",
            ],
            "DELETE /api/trading/orders": [
                "/api/portfolio/",
                "/api/trading/orders",
                "/api/risk/metrics",
            ],
            # Bot operations invalidate bot and portfolio data
            "POST /api/bots/start": [
                "/api/bots",
                "/api/portfolio/",
                "/api/monitoring/",
            ],
            "POST /api/bots/stop": [
                "/api/bots",
                "/api/portfolio/",
                "/api/monitoring/",
            ],
            # Configuration changes invalidate related data
            "PUT /api/risk/limits": [
                "/api/risk/",
                "/api/portfolio/",
            ],
            "POST /api/strategies/configure": [
                "/api/strategies/",
                "/api/bots",
            ],
        }

    async def dispatch(self, request: Request, call_next):
        """
        Process request through caching middleware.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response (cached or fresh)
        """
        # Check if caching should be bypassed
        if self._should_bypass_cache(request):
            response = await call_next(request)
            self._handle_cache_invalidation(request)
            return response

        # Generate cache key
        cache_key = self._generate_cache_key(request)

        # Check cache for GET requests
        if request.method == "GET":
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                self._add_cache_headers(cached_response, hit=True)
                return cached_response

        # Process request
        response = await call_next(request)

        # Cache successful GET responses
        if (
            request.method == "GET"
            and response.status_code == 200
            and self._should_cache_response(request, response)
        ):
            await self._cache_response(cache_key, request, response)

        # Handle cache invalidation for modifying operations
        self._handle_cache_invalidation(request)

        # Add cache headers
        self._add_cache_headers(response, hit=False)

        return response

    def _should_bypass_cache(self, request: Request) -> bool:
        """
        Check if caching should be bypassed for this request.

        Args:
            request: HTTP request

        Returns:
            True if cache should be bypassed
        """
        # Never cache non-GET requests
        if request.method != "GET":
            return True

        # Check no-cache patterns
        path = request.url.path
        for pattern in self.no_cache_patterns:
            if pattern in path:
                return True

        # Check for cache-bypass headers
        if request.headers.get("Cache-Control") == "no-cache":
            return True

        # Check for real-time data requests
        if "real-time" in request.query_params:
            return True

        return False

    def _should_cache_response(self, request: Request, response: Response) -> bool:
        """
        Check if response should be cached.

        Args:
            request: HTTP request
            response: HTTP response

        Returns:
            True if response should be cached
        """
        # Don't cache error responses
        if response.status_code >= 400:
            return False

        # Check if endpoint supports caching
        config = self._get_endpoint_config(request.url.path)
        return config.get("enabled", False)

    def _get_endpoint_config(self, path: str) -> dict[str, Any]:
        """
        Get caching configuration for endpoint.

        Args:
            path: Request path

        Returns:
            Cache configuration
        """
        # Exact match
        if path in self.cache_config:
            return self.cache_config[path]

        # Pattern matching
        for pattern, config in self.cache_config.items():
            if pattern.endswith("*"):
                if path.startswith(pattern[:-1]):
                    return config
            elif pattern in path:
                return config

        # Default configuration
        return {"ttl": self.default_ttl, "enabled": False}

    def _generate_cache_key(self, request: Request) -> str:
        """
        Generate cache key for request.

        Args:
            request: HTTP request

        Returns:
            Cache key string
        """
        # Include path, query parameters, and user context
        key_components = [
            request.method,
            request.url.path,
            str(sorted(request.query_params.items())),
        ]

        # Include user ID if authenticated
        if hasattr(request.state, "user"):
            user_data = getattr(request.state, "user", {})
            user_id = user_data.get("user_id", "anonymous")
            key_components.append(f"user:{user_id}")

        # Create hash of components
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Response | None:
        """
        Get cached response if available and valid.

        Args:
            cache_key: Cache key

        Returns:
            Cached response or None
        """
        if cache_key not in self.cache:
            return None

        entry = self.cache[cache_key]

        # Check if expired
        if entry.is_expired():
            del self.cache[cache_key]
            return None

        # Update access statistics
        entry.touch()

        # Create response from cached data
        return JSONResponse(
            content=entry.data,
            headers=entry.headers,
        )

    async def _cache_response(self, cache_key: str, request: Request, response: Response):
        """
        Cache response data.

        Args:
            cache_key: Cache key
            request: HTTP request
            response: HTTP response to cache
        """
        try:
            # Get TTL for this endpoint
            config = self._get_endpoint_config(request.url.path)
            ttl = config.get("ttl", self.default_ttl)

            # Extract response data
            if isinstance(response, JSONResponse):
                # For JSONResponse, get the content directly
                response_data = response.body
                if isinstance(response_data, bytes):
                    response_data = json.loads(response_data.decode())
                else:
                    response_data = response.body
            else:
                # For other response types, try to read body
                if hasattr(response, "body") and response.body:
                    try:
                        response_data = json.loads(response.body.decode())
                    except (json.JSONDecodeError, AttributeError):
                        return  # Don't cache non-JSON responses
                else:
                    return

            # Create cache entry
            expires_at = time.time() + ttl
            headers = dict(response.headers)

            entry = CacheEntry(data=response_data, headers=headers, expires_at=expires_at, path=request.url.path)

            # Evict old entries if cache is full
            if len(self.cache) >= self.max_cache_size:
                self._evict_lru_entries()

            # Store in cache
            self.cache[cache_key] = entry

            logger.debug(f"Cached response for {request.url.path} with TTL {ttl}s")

        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    def _evict_lru_entries(self):
        """Evict least recently used cache entries."""
        if not self.cache:
            return

        # Calculate how many entries to evict
        # For small caches, evict fewer entries to avoid over-eviction
        current_size = len(self.cache)
        if current_size <= 10:
            # For very small caches, evict only 1 entry at a time
            evict_count = 1
        else:
            # For larger caches, evict 25% but cap at current size
            evict_count = max(1, min(current_size, self.max_cache_size // 4))

        # Sort by last access time
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].last_access)

        # Evict oldest entries
        evicted_count = 0
        for i in range(evict_count):
            cache_key = sorted_entries[i][0]
            del self.cache[cache_key]
            evicted_count += 1

        logger.debug(f"Evicted {evicted_count} LRU cache entries")

    def _handle_cache_invalidation(self, request: Request):
        """
        Handle cache invalidation based on request.

        Args:
            request: HTTP request that might trigger invalidation
        """
        operation_key = f"{request.method} {request.url.path}"

        # Check for exact match invalidation triggers
        if operation_key in self.invalidation_triggers:
            patterns = self.invalidation_triggers[operation_key]
            self._invalidate_cache_patterns(patterns)
            return

        # Check for pattern-based invalidation
        for trigger_pattern, invalidate_patterns in self.invalidation_triggers.items():
            if self._matches_trigger_pattern(operation_key, trigger_pattern):
                self._invalidate_cache_patterns(invalidate_patterns)
                break

    def _matches_trigger_pattern(self, operation_key: str, pattern: str) -> bool:
        """Check if operation matches invalidation trigger pattern."""
        if "*" in pattern:
            pattern_prefix = pattern.split("*")[0]
            return operation_key.startswith(pattern_prefix)
        return operation_key == pattern

    def _invalidate_cache_patterns(self, patterns: list[str]):
        """
        Invalidate cache entries matching patterns.

        Args:
            patterns: List of path patterns to invalidate
        """
        keys_to_remove = []

        for cache_key, entry in self.cache.items():
            # Use the stored path for pattern matching
            for pattern in patterns:
                if pattern in entry.path:  # Match against the stored path
                    keys_to_remove.append(cache_key)
                    break

        # Remove invalidated entries
        for key in keys_to_remove:
            del self.cache[key]

        if keys_to_remove:
            logger.debug(f"Invalidated {len(keys_to_remove)} cache entries")

    def _add_cache_headers(self, response: Response, hit: bool = False):
        """
        Add cache-related headers to response.

        Args:
            response: HTTP response
            hit: Whether this was a cache hit
        """
        response.headers["X-Cache"] = "HIT" if hit else "MISS"
        response.headers["X-Cache-Time"] = str(int(time.time()))

        if not hit:
            # Add cache control headers for client-side caching
            response.headers["Cache-Control"] = "public, max-age=30"
            response.headers["ETag"] = f'"{hash(str(response.body)) % 2**32}"'

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache statistics
        """
        current_time = time.time()

        total_entries = len(self.cache)
        expired_entries = sum(1 for entry in self.cache.values() if entry.is_expired())
        total_hits = sum(entry.hit_count for entry in self.cache.values())

        # Calculate cache efficiency
        if total_entries > 0:
            avg_hits_per_entry = total_hits / total_entries
            avg_age = (
                sum(current_time - entry.created_at for entry in self.cache.values())
                / total_entries
                / 60
            )  # minutes
        else:
            avg_hits_per_entry = 0
            avg_age = 0

        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "cache_size_limit": self.max_cache_size,
            "cache_utilization": total_entries / self.max_cache_size,
            "total_hits": total_hits,
            "average_hits_per_entry": avg_hits_per_entry,
            "average_entry_age_minutes": avg_age,
            "cache_config_count": len(self.cache_config),
        }

    def clear_cache(self, pattern: str | None = None):
        """
        Clear cache entries.

        Args:
            pattern: Optional pattern to match for selective clearing
        """
        if pattern is None:
            # Clear all cache
            cleared_count = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared all {cleared_count} cache entries")
        else:
            # Clear matching entries
            keys_to_remove = [key for key in self.cache.keys() if pattern in key]

            for key in keys_to_remove:
                del self.cache[key]

            logger.info(f"Cleared {len(keys_to_remove)} cache entries matching '{pattern}'")
