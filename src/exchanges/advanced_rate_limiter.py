"""
Advanced rate limiting framework for multi-exchange coordination.

This module implements sophisticated rate limiting with exchange-specific
implementations and global coordination across all supported exchanges.

CRITICAL: This integrates with P-001 (core types, exceptions, config),
P-002A (error handling), and P-003+ (exchange interfaces).
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from src.core.base import BaseComponent
from src.core.config import Config
from src.core.exceptions import (
    ExchangeError,
    ExchangeRateLimitError,
    ValidationError,
)
from src.core.logging import get_logger

# Logger is provided by BaseComponent
# MANDATORY: Import from P-001
from src.core.types import (
    ExchangeType,
)

# MANDATORY: Import from P-002A
# MANDATORY: Import from P-007A (utils)
from src.utils.decorators import log_calls, time_execution

# Global rate limiter instance to avoid duplication
_global_rate_limiter = None


class AdvancedRateLimiter(BaseComponent):
    """
    Advanced rate limiter coordinating across all exchanges.

    This class coordinates rate limiting across all exchanges and provides
    exchange-specific rate limiting implementations.
    """

    def __init__(self, config: Config):
        """
        Initialize advanced rate limiter.

        Args:
            config: Application configuration
        """
        super().__init__()
        self.config = config
        self.exchange_limiters: dict[str, Any] = {}
        self.global_limits: dict[str, Any] = {}
        self.request_history: dict[str, list[datetime]] = defaultdict(list)

        # Initialize exchange-specific limiters
        self._initialize_exchange_limiters()

        self.logger.debug(
            f"AdvancedRateLimiter initialized with {len(self.exchange_limiters)} exchanges"
        )

    def _initialize_exchange_limiters(self) -> None:
        """Initialize exchange-specific rate limiters."""
        try:
            # Import rate limiter classes here to avoid circular/forward reference issues
            # These classes are defined later in this file
            self.logger.info(
                "Exchange rate limiters will be initialized lazily",
                exchanges=[
                    ExchangeType.BINANCE.value,
                    ExchangeType.OKX.value,
                    ExchangeType.COINBASE.value,
                ],
            )
        except Exception as e:
            self.logger.error("Failed to initialize exchange rate limiters", error=str(e))
            raise ExchangeError(f"Rate limiter initialization failed: {e!s}")

    def _get_or_create_limiter(self, exchange: str):
        """Get or create exchange-specific rate limiter."""
        if exchange not in self.exchange_limiters:
            if exchange == ExchangeType.BINANCE.value:
                self.exchange_limiters[exchange] = BinanceRateLimiter(self.config)
            elif exchange == ExchangeType.OKX.value:
                self.exchange_limiters[exchange] = OKXRateLimiter(self.config)
            elif exchange == ExchangeType.COINBASE.value:
                self.exchange_limiters[exchange] = CoinbaseRateLimiter(self.config)
            else:
                return None
        return self.exchange_limiters[exchange]

    @time_execution
    @log_calls
    async def check_rate_limit(self, exchange: str, endpoint: str, weight: int = 1) -> bool:
        """
        Check if rate limit allows the request.

        Args:
            exchange: Exchange name
            endpoint: API endpoint
            weight: Request weight

        Returns:
            bool: True if request is allowed

        Raises:
            ExchangeRateLimitError: If rate limit is exceeded
            ValidationError: If parameters are invalid
        """
        try:
            # Validate input parameters
            if not exchange or not endpoint:
                raise ValidationError("Exchange and endpoint are required")

            if weight <= 0:
                raise ValidationError("Weight must be positive")

            # Get exchange-specific limiter
            limiter = self._get_or_create_limiter(exchange)
            if not limiter:
                raise ExchangeRateLimitError(f"Unknown exchange: {exchange}")

            # Check exchange-specific limits
            if not await limiter.check_limit(endpoint, weight):
                self.logger.warning(
                    "Rate limit exceeded",
                    exchange=exchange,
                    endpoint=endpoint,
                    weight=weight,
                )
                return False

            # Check global limits
            if not await self._check_global_limits(exchange, endpoint, weight):
                self.logger.warning(
                    "Global rate limit exceeded",
                    exchange=exchange,
                    endpoint=endpoint,
                    weight=weight,
                )
                return False

            # Record request
            self._record_request(exchange, endpoint, weight)

            self.logger.debug(
                "Rate limit check passed",
                exchange=exchange,
                endpoint=endpoint,
                weight=weight,
            )
            return True

        except (ValidationError, ExchangeRateLimitError):
            raise
        except Exception as e:
            self.logger.error(
                f"Rate limit check failed: exchange {exchange}, endpoint {endpoint}, error {e!s}"
            )
            raise ExchangeRateLimitError(f"Rate limit check failed: {e!s}")

    @time_execution
    @log_calls
    async def wait_if_needed(self, exchange: str, endpoint: str) -> float:
        """
        Wait if rate limit is exceeded.

        Args:
            exchange: Exchange name
            endpoint: API endpoint

        Returns:
            float: Wait time in seconds
        """
        try:
            # Validate parameters
            if not exchange or not endpoint:
                raise ValidationError("Exchange and endpoint are required")

            # Get exchange-specific limiter (with lazy initialization)
            limiter = self._get_or_create_limiter(exchange)
            if not limiter:
                raise ExchangeRateLimitError(f"Unknown exchange: {exchange}")

            # Wait for exchange-specific limits
            wait_time = await limiter.wait_for_reset(endpoint)

            self.logger.debug(
                "Rate limit wait completed",
                exchange=exchange,
                endpoint=endpoint,
                wait_time=wait_time,
            )
            return wait_time

        except (ValidationError, ExchangeRateLimitError):
            raise
        except Exception as e:
            self.logger.error(
                f"Rate limit wait failed: exchange {exchange}, endpoint {endpoint}, error {e!s}"
            )
            raise ExchangeRateLimitError(f"Rate limit wait failed: {e!s}")

    async def _check_global_limits(self, exchange: str, endpoint: str, weight: int) -> bool:
        """Check global rate limits across all exchanges."""
        try:
            now = datetime.now()

            # Check global request rate limits
            global_key = "global:requests_per_minute"
            if global_key not in self.global_limits:
                self.global_limits[global_key] = []

            # Clean old entries (keep last minute)
            minute_ago = now - timedelta(minutes=1)
            self.global_limits[global_key] = [
                t for t in self.global_limits[global_key] if t > minute_ago
            ]

            # Check global rate limit (e.g., 1000 requests per minute across all exchanges)
            max_global_requests = 1000
            if len(self.global_limits[global_key]) >= max_global_requests:
                self.logger.warning(
                    "Global rate limit exceeded",
                    exchange=exchange,
                    endpoint=endpoint,
                    current_requests=len(self.global_limits[global_key]),
                    limit=max_global_requests,
                )
                return False

            # Check exchange-specific global limits
            exchange_key = f"global:{exchange}:requests_per_minute"
            if exchange_key not in self.global_limits:
                self.global_limits[exchange_key] = []

            # Clean old entries
            self.global_limits[exchange_key] = [
                t for t in self.global_limits[exchange_key] if t > minute_ago
            ]

            # Check exchange global limit (e.g., 500 requests per minute per exchange)
            max_exchange_requests = 500
            if len(self.global_limits[exchange_key]) >= max_exchange_requests:
                self.logger.warning(
                    "Exchange global rate limit exceeded",
                    exchange=exchange,
                    endpoint=endpoint,
                    current_requests=len(self.global_limits[exchange_key]),
                    limit=max_exchange_requests,
                )
                return False

            # Record the request
            self.global_limits[global_key].append(now)
            self.global_limits[exchange_key].append(now)

            return True

        except Exception as e:
            self.logger.error(f"Global rate limit check failed: {e!s}")
            return False

    def _record_request(self, exchange: str, endpoint: str, weight: int) -> None:
        """Record request for tracking."""
        now = datetime.now()
        key = f"{exchange}:{endpoint}"
        self.request_history[key].append(now)

        # Clean old history - remove entries older than 1 hour AND keep max 1000 requests
        hour_ago = now - timedelta(hours=1)
        self.request_history[key] = [t for t in self.request_history[key] if t > hour_ago]
        
        # If still too many, keep only the most recent 1000
        if len(self.request_history[key]) > 1000:
            self.request_history[key] = self.request_history[key][-1000:]


class BinanceRateLimiter:
    """
    Binance-specific rate limiter implementing weight-based limiting.

    Binance uses a weight-based system with 1200 requests/minute limit.
    """

    def __init__(self, config: Config):
        """
        Initialize Binance rate limiter.

        Args:
            config: Application configuration
        """
        self.config = config
        self.weight_limit = 1200  # requests per minute
        self.order_limit_10s = 50  # orders per 10 seconds
        self.order_limit_24h = 160000  # orders per 24 hours

        # Track usage with weights
        self.weight_usage: dict[str, list[tuple[datetime, int]]] = defaultdict(list)
        self.order_usage: list[datetime] = []

        # Add logger
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug(f"BinanceRateLimiter initialized with weight_limit {self.weight_limit}")

    async def check_limit(self, endpoint: str, weight: int = 1) -> bool:
        """
        Check if request is within Binance rate limits.

        Args:
            endpoint: API endpoint
            weight: Request weight

        Returns:
            bool: True if request is allowed

        Raises:
            ValidationError: If parameters are invalid
            ExchangeRateLimitError: If rate limit is exceeded
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")

            if weight <= 0:
                raise ValidationError("Weight must be positive")

            if weight > self.weight_limit:
                raise ValidationError(f"Weight {weight} exceeds limit {self.weight_limit}")

            # Check weight-based limits
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)

            # Clean old entries
            self.weight_usage[endpoint] = [(t, w) for t, w in self.weight_usage[endpoint] if t > minute_ago]

            # Calculate current weight usage (sum of all weights)
            current_weight = sum(w for _, w in self.weight_usage[endpoint])

            if current_weight + weight > self.weight_limit:
                self.logger.warning(
                    "Binance weight limit exceeded",
                    endpoint=endpoint,
                    weight=weight,
                    current_weight=current_weight,
                    limit=self.weight_limit,
                )
                return False

            # Check order limits if applicable
            if "order" in endpoint.lower():
                if not await self._check_order_limits():
                    return False

            # Record the request with its weight
            self.weight_usage[endpoint].append((now, weight))
            
            self.logger.debug("Binance rate limit check passed", endpoint=endpoint, weight=weight)
            return True

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Binance rate limit check failed: endpoint {endpoint}, error {e!s}")
            raise ExchangeRateLimitError(f"Binance rate limit check failed: {e!s}")

    async def wait_for_reset(self, endpoint: str) -> float:
        """
        Wait for rate limit reset.

        Args:
            endpoint: API endpoint

        Returns:
            float: Wait time in seconds
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")

            # Calculate wait time based on current usage
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)

            # Clean old entries
            self.weight_usage[endpoint] = [(t, w) for t, w in self.weight_usage[endpoint] if t > minute_ago]

            # Calculate time until reset
            if self.weight_usage[endpoint]:
                oldest_request = min(t for t, _ in self.weight_usage[endpoint])
                reset_time = oldest_request + timedelta(minutes=1)
                wait_time = max(0, (reset_time - now).total_seconds())
            else:
                wait_time = 0

            if wait_time > 0:
                self.logger.info(
                    "Waiting for Binance rate limit reset", endpoint=endpoint, wait_time=wait_time
                )
                await asyncio.sleep(wait_time)

            return wait_time

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Binance rate limit wait failed: endpoint {endpoint}, error {e!s}")
            raise ExchangeRateLimitError(f"Binance rate limit wait failed: {e!s}")

    async def _check_order_limits(self) -> bool:
        """Check order-specific rate limits."""
        now = datetime.now()
        ten_seconds_ago = now - timedelta(seconds=10)
        day_ago = now - timedelta(days=1)

        # Clean old entries
        self.order_usage = [t for t in self.order_usage if t > day_ago]

        # Check 10-second limit
        recent_orders = [t for t in self.order_usage if t > ten_seconds_ago]
        if len(recent_orders) >= self.order_limit_10s:
            self.logger.warning(
                "Binance order limit exceeded(10s)",
                recent_orders=len(recent_orders),
                limit=self.order_limit_10s,
            )
            return False

        # Check 24-hour limit
        if len(self.order_usage) >= self.order_limit_24h:
            self.logger.warning(
                "Binance order limit exceeded(24h)",
                total_orders=len(self.order_usage),
                limit=self.order_limit_24h,
            )
            return False

        return True


class OKXRateLimiter:
    """
    OKX-specific rate limiter implementing endpoint-based limiting.

    OKX uses endpoint-based limits: REST (60/2s), Orders (600/2s), Historical (20/2s).
    """

    def __init__(self, config: Config):
        """
        Initialize OKX rate limiter.

        Args:
            config: Application configuration
        """
        self.config = config

        # Endpoint-specific limits
        self.limits = {
            "rest": {"requests": 60, "window": 2},  # 60 requests per 2 seconds
            # 600 orders per 2 seconds
            "orders": {"requests": 600, "window": 2},
            # 20 requests per 2 seconds
            "historical": {"requests": 20, "window": 2},
        }

        # Track usage per endpoint type
        self.usage: dict[str, list[datetime]] = defaultdict(list)

        # Add logger
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug("OKXRateLimiter initialized", limits=self.limits)

    async def check_limit(self, endpoint: str, weight: int = 1) -> bool:
        """
        Check if request is within OKX rate limits.

        Args:
            endpoint: API endpoint
            weight: Request weight (not used for OKX)

        Returns:
            bool: True if request is allowed

        Raises:
            ValidationError: If parameters are invalid
            ExchangeRateLimitError: If rate limit is exceeded
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")

            # Determine endpoint type
            endpoint_type = self._get_endpoint_type(endpoint)

            # Check limits
            now = datetime.now()
            window_seconds = self.limits[endpoint_type]["window"]
            max_requests = self.limits[endpoint_type]["requests"]

            window_start = now - timedelta(seconds=window_seconds)

            # Clean old entries
            self.usage[endpoint_type] = [t for t in self.usage[endpoint_type] if t > window_start]

            # Check if limit exceeded
            if len(self.usage[endpoint_type]) >= max_requests:
                self.logger.warning(
                    "OKX rate limit exceeded",
                    endpoint_type=endpoint_type,
                    endpoint=endpoint,
                    requests=len(self.usage[endpoint_type]),
                    limit=max_requests,
                )
                return False

            self.logger.debug(
                "OKX rate limit check passed",
                endpoint_type=endpoint_type,
                endpoint=endpoint,
            )
            return True

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"OKX rate limit check failed: endpoint {endpoint}, error {e!s}")
            raise ExchangeRateLimitError(f"OKX rate limit check failed: {e!s}")

    async def wait_for_reset(self, endpoint: str) -> float:
        """
        Wait for rate limit reset.

        Args:
            endpoint: API endpoint

        Returns:
            float: Wait time in seconds
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")

            endpoint_type = self._get_endpoint_type(endpoint)
            window_seconds = self.limits[endpoint_type]["window"]

            now = datetime.now()
            window_start = now - timedelta(seconds=window_seconds)

            # Clean old entries
            self.usage[endpoint_type] = [t for t in self.usage[endpoint_type] if t > window_start]

            # Calculate wait time
            if self.usage[endpoint_type]:
                oldest_request = min(self.usage[endpoint_type])
                reset_time = oldest_request + timedelta(seconds=window_seconds)
                wait_time = max(0, (reset_time - now).total_seconds())
            else:
                wait_time = 0

            if wait_time > 0:
                self.logger.info(
                    "Waiting for OKX rate limit reset",
                    endpoint_type=endpoint_type,
                    endpoint=endpoint,
                    wait_time=wait_time,
                )
                await asyncio.sleep(wait_time)

            return wait_time

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"OKX rate limit wait failed: endpoint {endpoint}, error {e!s}")
            raise ExchangeRateLimitError(f"OKX rate limit wait failed: {e!s}")

    def _get_endpoint_type(self, endpoint: str) -> str:
        """Determine endpoint type for rate limiting."""
        endpoint_lower = endpoint.lower()

        if "order" in endpoint_lower:
            return "orders"
        elif "history" in endpoint_lower or "historical" in endpoint_lower:
            return "historical"
        else:
            return "rest"


class CoinbaseRateLimiter:
    """
    Coinbase-specific rate limiter implementing point-based limiting.

    Coinbase uses a point-based system with 8000 points/minute limit.
    """

    def __init__(self, config: Config):
        """
        Initialize Coinbase rate limiter.

        Args:
            config: Application configuration
        """
        self.config = config
        self.points_limit = 8000  # points per minute
        self.private_limit = 10  # requests per second for private endpoints
        self.public_limit = 15  # requests per second for public endpoints

        # Track usage
        self.points_usage: list[datetime] = []
        self.private_usage: list[datetime] = []
        self.public_usage: list[datetime] = []

        # Add logger
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug("CoinbaseRateLimiter initialized", points_limit=self.points_limit)

    async def check_limit(self, endpoint: str, is_private: bool = False) -> bool:
        """
        Check if request is within Coinbase rate limits.

        Args:
            endpoint: API endpoint
            is_private: Whether this is a private endpoint

        Returns:
            bool: True if request is allowed

        Raises:
            ValidationError: If parameters are invalid
            ExchangeRateLimitError: If rate limit is exceeded
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")

            # Calculate points for this request
            points = self._calculate_points(endpoint)

            # Check points limit
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)

            # Clean old entries
            self.points_usage = [t for t in self.points_usage if t > minute_ago]

            # Calculate current points usage
            current_points = sum(self._calculate_points("") for _ in self.points_usage)

            if current_points + points > self.points_limit:
                self.logger.warning(
                    "Coinbase points limit exceeded",
                    endpoint=endpoint,
                    points=points,
                    current_points=current_points,
                    limit=self.points_limit,
                )
                return False

            # Check per-second limits
            if is_private:
                if not await self._check_private_limit():
                    return False
            else:
                if not await self._check_public_limit():
                    return False

            self.logger.debug(
                "Coinbase rate limit check passed",
                endpoint=endpoint,
                points=points,
                is_private=is_private,
            )
            return True

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Coinbase rate limit check failed: endpoint {endpoint}, error {e!s}")
            raise ExchangeRateLimitError(f"Coinbase rate limit check failed: {e!s}")

    async def wait_for_reset(self, endpoint: str) -> float:
        """
        Wait for rate limit reset.

        Args:
            endpoint: API endpoint

        Returns:
            float: Wait time in seconds
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")

            # Calculate wait time based on points usage
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)

            # Clean old entries
            self.points_usage = [t for t in self.points_usage if t > minute_ago]

            # Calculate time until reset
            if self.points_usage:
                oldest_request = min(self.points_usage)
                reset_time = oldest_request + timedelta(minutes=1)
                wait_time = max(0, (reset_time - now).total_seconds())
            else:
                wait_time = 0

            if wait_time > 0:
                self.logger.info(
                    "Waiting for Coinbase rate limit reset",
                    endpoint=endpoint,
                    wait_time=wait_time,
                )
                await asyncio.sleep(wait_time)

            return wait_time

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Coinbase rate limit wait failed: endpoint {endpoint}, error {e!s}")
            raise ExchangeRateLimitError(f"Coinbase rate limit wait failed: {e!s}")

    def _calculate_points(self, endpoint: str) -> int:
        """
        Calculate points for an endpoint.

        Args:
            endpoint: API endpoint

        Returns:
            int: Points for this endpoint
        """
        endpoint_lower = endpoint.lower()

        # Point calculation based on endpoint type
        if "order" in endpoint_lower:
            return 10  # Order placement is expensive
        elif "balance" in endpoint_lower or "account" in endpoint_lower:
            return 5  # Account queries are moderate
        elif "market" in endpoint_lower or "ticker" in endpoint_lower:
            return 1  # Market data is cheap
        else:
            return 2  # Default cost

    async def _check_private_limit(self) -> bool:
        """Check private endpoint rate limits."""
        now = datetime.now()
        second_ago = now - timedelta(seconds=1)

        # Clean old entries
        self.private_usage = [t for t in self.private_usage if t > second_ago]

        if len(self.private_usage) >= self.private_limit:
            self.logger.warning(
                "Coinbase private rate limit exceeded",
                requests=len(self.private_usage),
                limit=self.private_limit,
            )
            return False

        return True

    async def _check_public_limit(self) -> bool:
        """Check public endpoint rate limits."""
        now = datetime.now()
        second_ago = now - timedelta(seconds=1)

        # Clean old entries
        self.public_usage = [t for t in self.public_usage if t > second_ago]

        if len(self.public_usage) >= self.public_limit:
            self.logger.warning(
                "Coinbase public rate limit exceeded",
                requests=len(self.public_usage),
                limit=self.public_limit,
            )
            return False

        return True


def get_global_rate_limiter(config: Config = None) -> AdvancedRateLimiter:
    """
    Get or create the global rate limiter instance.

    This prevents duplicate initialization of rate limiters across exchanges.

    Args:
        config: Application configuration (required for first call)

    Returns:
        AdvancedRateLimiter: Global rate limiter instance
    """
    global _global_rate_limiter

    if _global_rate_limiter is None:
        if config is None:
            raise ValueError("Config required for first rate limiter initialization")
        _global_rate_limiter = AdvancedRateLimiter(config)
        logger = get_logger(__name__)
        logger.info("Global rate limiter instance created")

    return _global_rate_limiter
