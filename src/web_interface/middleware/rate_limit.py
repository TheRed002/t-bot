"""
Rate limiting middleware for T-Bot web interface.

This middleware provides comprehensive rate limiting with different tiers
for different user types and endpoint categories.
"""

import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import Config
from src.core.logging import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting middleware.

    This middleware provides:
    - Per-user and per-IP rate limiting
    - Different limits for different user tiers
    - Endpoint-specific rate limits
    - Sliding window rate limiting
    - Rate limit headers in responses
    """

    def __init__(self, app, config: Config):
        """
        Initialize rate limiting middleware.

        Args:
            app: FastAPI application
            config: Application configuration
        """
        super().__init__(app)
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Rate limit configuration - handle missing web_interface config
        if hasattr(config, "web_interface"):
            rate_limit_config = config.web_interface.get("rate_limiting", {})
        else:
            # Default rate limiting config
            rate_limit_config = {
                "enabled": True,  # Rate limiting enabled by default
                "anonymous_limit": 60,
                "authenticated_limit": 300,
                "trader_limit": 600,
                "admin_limit": 1200,
                "trading_limit": 120,
                "bot_action_limit": 30,
            }

        # Check if rate limiting is enabled
        self.enabled = rate_limit_config.get("enabled", True)

        # In test environment, check for explicit disable
        if hasattr(config, "environment") and config.environment == "test":
            # Allow disabling rate limiting in test mode
            self.enabled = rate_limit_config.get("enabled", True)
            if not self.enabled:
                self.logger.info("Rate limiting disabled for testing environment")

        # Default rate limits (requests per minute)
        self.default_limits = {
            "anonymous": rate_limit_config.get("anonymous_limit", 10),
            "authenticated": rate_limit_config.get("authenticated_limit", 60),
            "trader": rate_limit_config.get("trader_limit", 120),
            "admin": rate_limit_config.get("admin_limit", 300),
        }

        # Endpoint-specific limits (requests per minute)
        self.endpoint_limits = {
            "/api/trading/orders": rate_limit_config.get("trading_limit", 30),
            "/api/bots/start": rate_limit_config.get("bot_action_limit", 10),
            "/api/bots/stop": rate_limit_config.get("bot_action_limit", 10),
            "/auth/login": rate_limit_config.get("auth_limit", 5),
            "/auth/refresh": rate_limit_config.get("auth_limit", 10),
        }

        # Sliding window configuration
        self.window_size_seconds = rate_limit_config.get("window_size_seconds", 60)
        self.cleanup_interval = rate_limit_config.get("cleanup_interval", 300)  # 5 minutes

        # Request tracking
        self.request_counts: dict[str, deque] = defaultdict(deque)
        self.last_cleanup = time.time()

        # Exempted paths
        self.exempt_paths = {"/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through rate limiting middleware.

        Args:
            request: HTTP request
            call_next: Next middleware/endpoint

        Returns:
            Response: HTTP response

        Raises:
            HTTPException: If rate limit exceeded
        """
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)

        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Clean up old entries periodically
        await self._cleanup_old_entries()

        # Determine rate limit key and limits
        rate_limit_key, rate_limit = self._get_rate_limit_info(request)

        # Check rate limit
        current_time = time.time()
        if not self._check_rate_limit(rate_limit_key, rate_limit, current_time):
            # Rate limit exceeded
            retry_after = self._calculate_retry_after(rate_limit_key, current_time)

            self.logger.warning(
                "Rate limit exceeded",
                key=rate_limit_key,
                path=request.url.path,
                method=request.method,
                limit=rate_limit,
                retry_after=retry_after,
            )

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(rate_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + retry_after)),
                },
            )

        # Record request
        self._record_request(rate_limit_key, current_time)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        self._add_rate_limit_headers(response, rate_limit_key, rate_limit)

        return response

    def _get_rate_limit_info(self, request: Request) -> tuple[str, int]:
        """
        Determine rate limit key and limit for request.

        Args:
            request: HTTP request

        Returns:
            tuple: (rate_limit_key, rate_limit)
        """
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"

        # Check if user is authenticated
        if hasattr(request.state, "authenticated") and request.state.authenticated:
            user_data = getattr(request.state, "user", {})
            user_id = user_data.get("user_id", "unknown")
            scopes = user_data.get("scopes", [])

            # Determine user tier
            if "admin" in scopes:
                user_tier = "admin"
            elif "trade" in scopes:
                user_tier = "trader"
            else:
                user_tier = "authenticated"

            rate_limit_key = f"user:{user_id}"
            base_limit = self.default_limits[user_tier]
        else:
            rate_limit_key = f"ip:{client_ip}"
            base_limit = self.default_limits["anonymous"]

        # Check for endpoint-specific limits
        endpoint_limit = self._get_endpoint_limit(request.url.path)
        rate_limit = min(base_limit, endpoint_limit) if endpoint_limit else base_limit

        return rate_limit_key, rate_limit

    def _get_endpoint_limit(self, path: str) -> int | None:
        """
        Get endpoint-specific rate limit.

        Args:
            path: Request path

        Returns:
            int: Endpoint rate limit or None
        """
        # Exact match
        if path in self.endpoint_limits:
            return self.endpoint_limits[path]

        # Pattern matching for API endpoints
        for endpoint_pattern, limit in self.endpoint_limits.items():
            if path.startswith(endpoint_pattern.rstrip("*")):
                return limit

        return None

    def _check_rate_limit(self, key: str, limit: int, current_time: float) -> bool:
        """
        Check if request is within rate limit.

        Args:
            key: Rate limit key
            limit: Rate limit (requests per window)
            current_time: Current timestamp

        Returns:
            bool: True if within limit
        """
        # Get request timestamps for this key
        request_times = self.request_counts[key]

        # Remove old requests outside the window
        window_start = current_time - self.window_size_seconds
        while request_times and request_times[0] < window_start:
            request_times.popleft()

        # Check if we're within the limit
        return len(request_times) < limit

    def _record_request(self, key: str, current_time: float) -> None:
        """
        Record a request for rate limiting.

        Args:
            key: Rate limit key
            current_time: Current timestamp
        """
        self.request_counts[key].append(current_time)

    def _calculate_retry_after(self, key: str, current_time: float) -> int:
        """
        Calculate retry-after seconds.

        Args:
            key: Rate limit key
            current_time: Current timestamp

        Returns:
            int: Seconds to wait before retry
        """
        request_times = self.request_counts.get(key, deque())
        if not request_times:
            return 60  # Default retry after 1 minute

        # Time until oldest request falls outside the window
        oldest_request = request_times[0]
        time_until_reset = (oldest_request + self.window_size_seconds) - current_time

        return max(1, int(time_until_reset))

    def _add_rate_limit_headers(self, response: Response, key: str, limit: int) -> None:
        """
        Add rate limit headers to response.

        Args:
            response: HTTP response
            key: Rate limit key
            limit: Rate limit
        """
        request_times = self.request_counts.get(key, deque())
        current_time = time.time()

        # Calculate remaining requests
        window_start = current_time - self.window_size_seconds
        recent_requests = sum(1 for t in request_times if t >= window_start)
        remaining = max(0, limit - recent_requests)

        # Calculate reset time
        reset_time = int(current_time + self.window_size_seconds)

        # Add headers
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        response.headers["X-RateLimit-Window"] = str(self.window_size_seconds)

    async def _cleanup_old_entries(self) -> None:
        """Clean up old rate limit entries."""
        current_time = time.time()

        # Only cleanup periodically
        if current_time - self.last_cleanup < self.cleanup_interval:
            return

        self.last_cleanup = current_time
        window_start = current_time - self.window_size_seconds

        # Clean up old entries
        keys_to_remove = []
        for key, request_times in self.request_counts.items():
            # Remove old requests
            while request_times and request_times[0] < window_start:
                request_times.popleft()

            # Remove empty keys
            if not request_times:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.request_counts[key]

        if keys_to_remove:
            self.logger.debug(f"Cleaned up {len(keys_to_remove)} expired rate limit entries")

    def get_rate_limit_stats(self) -> dict:
        """
        Get rate limiting statistics.

        Returns:
            dict: Rate limiting statistics
        """
        current_time = time.time()
        window_start = current_time - self.window_size_seconds

        active_keys = 0
        total_recent_requests = 0

        for _key, request_times in self.request_counts.items():
            recent_requests = sum(1 for t in request_times if t >= window_start)
            if recent_requests > 0:
                active_keys += 1
                total_recent_requests += recent_requests

        return {
            "window_size_seconds": self.window_size_seconds,
            "active_keys": active_keys,
            "total_tracked_keys": len(self.request_counts),
            "total_recent_requests": total_recent_requests,
            "default_limits": self.default_limits,
            "endpoint_limits": self.endpoint_limits,
            "last_cleanup": self.last_cleanup,
        }
