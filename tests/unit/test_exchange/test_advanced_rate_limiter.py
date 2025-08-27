"""
Unit tests for advanced rate limiting framework.

This module tests the advanced rate limiting functionality including
exchange-specific rate limiters and global coordination.

CRITICAL: Tests must achieve 90% coverage for P-007 components.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.config import Config
from src.core.exceptions import ExchangeRateLimitError, ValidationError
from src.core.types import ExchangeType, RequestType

# Import the components to test
from src.exchanges.advanced_rate_limiter import (
    AdvancedRateLimiter,
    BinanceRateLimiter,
    CoinbaseRateLimiter,
    OKXRateLimiter,
)


class TestAdvancedRateLimiter:
    """Test cases for AdvancedRateLimiter class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Mock(spec=Config)

    @pytest.fixture
    def rate_limiter(self, config):
        """Create an AdvancedRateLimiter instance."""
        return AdvancedRateLimiter(config)

    def test_initialization(self, rate_limiter):
        """Test that the rate limiter initializes correctly."""
        assert rate_limiter is not None
        assert hasattr(rate_limiter, "exchange_limiters")
        assert hasattr(rate_limiter, "global_limits")
        assert hasattr(rate_limiter, "request_history")

        # Exchange limiters use lazy initialization, so they should be empty initially
        assert len(rate_limiter.exchange_limiters) == 0
        
        # Test that limiters can be created on demand
        expected_exchanges = ["binance", "okx", "coinbase"]
        for exchange in expected_exchanges:
            limiter = rate_limiter._get_or_create_limiter(exchange)
            assert limiter is not None
            assert exchange in rate_limiter.exchange_limiters

    @pytest.mark.asyncio
    async def test_check_rate_limit_valid_request(self, rate_limiter):
        """Test rate limit check with valid request."""
        # Create a mock limiter and mock the _get_or_create_limiter method
        mock_limiter = AsyncMock()
        mock_limiter.check_limit.return_value = True
        
        with (
            patch.object(rate_limiter, "_get_or_create_limiter", return_value=mock_limiter),
            patch.object(rate_limiter, "_check_global_limits", return_value=True)
        ):
            result = await rate_limiter.check_rate_limit("binance", "/api/v3/ticker/price", 1)
            assert result is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_invalid_exchange(self, rate_limiter):
        """Test rate limit check with invalid exchange."""
        with pytest.raises(ExchangeRateLimitError, match="Unknown exchange"):
            await rate_limiter.check_rate_limit("invalid_exchange", "/api/v3/ticker/price", 1)

    @pytest.mark.asyncio
    async def test_check_rate_limit_invalid_endpoint(self, rate_limiter):
        """Test rate limit check with invalid endpoint."""
        with pytest.raises(ValidationError, match="Exchange and endpoint are required"):
            await rate_limiter.check_rate_limit("binance", "", 1)

    @pytest.mark.asyncio
    async def test_check_rate_limit_invalid_weight(self, rate_limiter):
        """Test rate limit check with invalid weight."""
        with pytest.raises(ValidationError, match="Weight must be positive"):
            await rate_limiter.check_rate_limit("binance", "/api/v3/ticker/price", 0)

    @pytest.mark.asyncio
    async def test_wait_if_needed_valid_request(self, rate_limiter):
        """Test wait if needed with valid request."""
        # Create a mock limiter and mock the _get_or_create_limiter method
        mock_limiter = AsyncMock()
        mock_limiter.wait_for_reset.return_value = 0.0
        
        with patch.object(rate_limiter, "_get_or_create_limiter", return_value=mock_limiter):
            result = await rate_limiter.wait_if_needed("binance", "/api/v3/ticker/price")
            assert result == 0.0

    @pytest.mark.asyncio
    async def test_wait_if_needed_invalid_exchange(self, rate_limiter):
        """Test wait if needed with invalid exchange."""
        with pytest.raises(ExchangeRateLimitError, match="Unknown exchange"):
            await rate_limiter.wait_if_needed("invalid_exchange", "/api/v3/ticker/price")

    @pytest.mark.asyncio
    async def test_wait_if_needed_invalid_endpoint(self, rate_limiter):
        """Test wait if needed with invalid endpoint."""
        with pytest.raises(ValidationError, match="Exchange and endpoint are required"):
            await rate_limiter.wait_if_needed("binance", "")

    def test_record_request(self, rate_limiter):
        """Test recording a request."""
        exchange = "binance"
        endpoint = "/api/v3/ticker/price"
        weight = 1

        rate_limiter._record_request(exchange, endpoint, weight)

        key = f"{exchange}:{endpoint}"
        assert key in rate_limiter.request_history
        assert len(rate_limiter.request_history[key]) == 1

    @pytest.mark.asyncio
    async def test_check_rate_limit_exchange_specific_limit_exceeded(self, rate_limiter):
        """Test rate limit check when exchange-specific limit is exceeded."""
        # Create a mock limiter that returns False
        mock_limiter = AsyncMock()
        mock_limiter.check_limit.return_value = False
        
        with patch.object(rate_limiter, "_get_or_create_limiter", return_value=mock_limiter):
            result = await rate_limiter.check_rate_limit("binance", "/api/v3/ticker/price", 1)
            assert result is False

    @pytest.mark.asyncio
    async def test_check_rate_limit_global_limit_exceeded(self, rate_limiter):
        """Test rate limit check when global limit is exceeded."""
        # Create a mock limiter that returns True
        mock_limiter = AsyncMock()
        mock_limiter.check_limit.return_value = True
        
        with (
            patch.object(rate_limiter, "_get_or_create_limiter", return_value=mock_limiter),
            patch.object(rate_limiter, "_check_global_limits", return_value=False)
        ):
            result = await rate_limiter.check_rate_limit("binance", "/api/v3/ticker/price", 1)
            assert result is False

    @pytest.mark.asyncio
    async def test_check_rate_limit_exception_handling(self, rate_limiter):
        """Test rate limit check with exception handling."""
        # Create a mock limiter that raises an exception
        mock_limiter = AsyncMock()
        mock_limiter.check_limit.side_effect = Exception("Test error")
        
        with patch.object(rate_limiter, "_get_or_create_limiter", return_value=mock_limiter):
            with pytest.raises(ExchangeRateLimitError):
                await rate_limiter.check_rate_limit("binance", "/api/v3/ticker/price", 1)

    @pytest.mark.asyncio
    async def test_wait_if_needed_exception_handling(self, rate_limiter):
        """Test wait_if_needed with exception handling."""
        # Create a mock limiter that raises an exception
        mock_limiter = AsyncMock()
        mock_limiter.wait_for_reset.side_effect = Exception("Test error")
        
        with patch.object(rate_limiter, "_get_or_create_limiter", return_value=mock_limiter):
            with pytest.raises(ExchangeRateLimitError):
                await rate_limiter.wait_if_needed("binance", "/api/v3/ticker/price")

    @pytest.mark.asyncio
    async def test_check_global_limits(self, rate_limiter):
        """Test global limits checking."""
        result = await rate_limiter._check_global_limits("binance", "/api/v3/ticker/price", 1)
        assert result is True  # Currently always returns True

    def test_record_request_cleanup(self, rate_limiter):
        """Test request recording with cleanup."""
        # Add more than 1000 requests to trigger cleanup
        now = datetime.now()
        for _ in range(1001):
            rate_limiter._record_request("binance", "/api/v3/ticker/price", 1)

        # Check that cleanup worked (should have 1000 or fewer entries)
        key = "binance:/api/v3/ticker/price"
        assert len(rate_limiter.request_history[key]) <= 1000


class TestBinanceRateLimiter:
    """Test cases for BinanceRateLimiter class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Mock(spec=Config)

    @pytest.fixture
    def rate_limiter(self, config):
        """Create a BinanceRateLimiter instance."""
        return BinanceRateLimiter(config)

    def test_initialization(self, rate_limiter):
        """Test that the rate limiter initializes correctly."""
        assert rate_limiter.weight_limit == 1200
        assert rate_limiter.order_limit_10s == 50
        assert rate_limiter.order_limit_24h == 160000
        assert hasattr(rate_limiter, "weight_usage")
        assert hasattr(rate_limiter, "order_usage")

    @pytest.mark.asyncio
    async def test_check_limit_valid_request(self, rate_limiter):
        """Test rate limit check with valid request."""
        result = await rate_limiter.check_limit("/api/v3/ticker/price", 1)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_limit_invalid_endpoint(self, rate_limiter):
        """Test rate limit check with invalid endpoint."""
        with pytest.raises(ValidationError, match="Endpoint is required"):
            await rate_limiter.check_limit("", 1)

    @pytest.mark.asyncio
    async def test_check_limit_invalid_weight(self, rate_limiter):
        """Test rate limit check with invalid weight."""
        with pytest.raises(ValidationError, match="Weight must be positive"):
            await rate_limiter.check_limit("/api/v3/ticker/price", 0)

    @pytest.mark.asyncio
    async def test_check_limit_weight_exceeds_limit(self, rate_limiter):
        """Test rate limit check with weight exceeding limit."""
        with pytest.raises(ValidationError, match="Weight 1201 exceeds limit 1200"):
            await rate_limiter.check_limit("/api/v3/ticker/price", 1201)

    @pytest.mark.asyncio
    async def test_wait_for_reset_valid_request(self, rate_limiter):
        """Test wait for reset with valid request."""
        result = await rate_limiter.wait_for_reset("/api/v3/ticker/price")
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_wait_for_reset_invalid_endpoint(self, rate_limiter):
        """Test wait for reset with invalid endpoint."""
        with pytest.raises(ValidationError, match="Endpoint is required"):
            await rate_limiter.wait_for_reset("")

    @pytest.mark.asyncio
    async def test_check_order_limits_valid(self, rate_limiter):
        """Test order limits check with valid state."""
        result = await rate_limiter._check_order_limits()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_order_limits_exceeded_10s(self, rate_limiter):
        """Test order limits check when 10s limit is exceeded."""
        # Add more than 50 orders in the last 10 seconds
        now = datetime.now()
        for _ in range(51):
            rate_limiter.order_usage.append(now)

        result = await rate_limiter._check_order_limits()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_order_limits_exceeded_24h(self, rate_limiter):
        """Test order limits check when 24h limit is exceeded."""
        # Add more than 160000 orders
        now = datetime.now()
        for _ in range(160001):
            rate_limiter.order_usage.append(now)

        result = await rate_limiter._check_order_limits()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_limit_exception_handling(self, rate_limiter):
        """Test check_limit with exception handling."""
        # Mock _check_order_limits to raise an exception
        rate_limiter._check_order_limits = AsyncMock(side_effect=Exception("Test error"))

        with pytest.raises(ExchangeRateLimitError):
            await rate_limiter.check_limit("/api/v3/order", 1)

    @pytest.mark.asyncio
    async def test_wait_for_reset_exception_handling(self, rate_limiter):
        """Test wait_for_reset with exception handling."""
        # Mock _check_order_limits to raise an exception
        rate_limiter._check_order_limits = AsyncMock(side_effect=Exception("Test error"))

        # The wait_for_reset method doesn't call _check_order_limits, so we need to mock something else
        # Let's mock the weight_usage to cause an exception
        rate_limiter.weight_usage = None  # This will cause an exception

        with pytest.raises(ExchangeRateLimitError):
            await rate_limiter.wait_for_reset("/api/v3/order")

    @pytest.mark.asyncio
    async def test_check_order_limits_exception_handling(self, rate_limiter):
        """Test _check_order_limits with exception handling."""
        # Mock to raise an exception
        rate_limiter.order_usage = None  # This will cause an exception

        with pytest.raises(Exception):
            await rate_limiter._check_order_limits()


class TestOKXRateLimiter:
    """Test cases for OKXRateLimiter class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Mock(spec=Config)

    @pytest.fixture
    def rate_limiter(self, config):
        """Create an OKXRateLimiter instance."""
        return OKXRateLimiter(config)

    def test_initialization(self, rate_limiter):
        """Test that the rate limiter initializes correctly."""
        assert "rest" in rate_limiter.limits
        assert "orders" in rate_limiter.limits
        assert "historical" in rate_limiter.limits
        assert rate_limiter.limits["rest"]["requests"] == 60
        assert rate_limiter.limits["rest"]["window"] == 2

    @pytest.mark.asyncio
    async def test_check_limit_valid_request(self, rate_limiter):
        """Test rate limit check with valid request."""
        result = await rate_limiter.check_limit("/api/v5/market/ticker", 1)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_limit_invalid_endpoint(self, rate_limiter):
        """Test rate limit check with invalid endpoint."""
        with pytest.raises(ValidationError, match="Endpoint is required"):
            await rate_limiter.check_limit("", 1)

    @pytest.mark.asyncio
    async def test_wait_for_reset_valid_request(self, rate_limiter):
        """Test wait for reset with valid request."""
        result = await rate_limiter.wait_for_reset("/api/v5/market/ticker")
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_wait_for_reset_invalid_endpoint(self, rate_limiter):
        """Test wait for reset with invalid endpoint."""
        with pytest.raises(ValidationError, match="Endpoint is required"):
            await rate_limiter.wait_for_reset("")

    def test_get_endpoint_type_rest(self, rate_limiter):
        """Test endpoint type detection for REST endpoints."""
        endpoint_type = rate_limiter._get_endpoint_type("/api/v5/market/ticker")
        assert endpoint_type == "rest"

    def test_get_endpoint_type_orders(self, rate_limiter):
        """Test endpoint type detection for order endpoints."""
        endpoint_type = rate_limiter._get_endpoint_type("/api/v5/trade/order")
        assert endpoint_type == "orders"

    def test_get_endpoint_type_historical(self, rate_limiter):
        """Test endpoint type detection for historical endpoints."""
        endpoint_type = rate_limiter._get_endpoint_type("/api/v5/market/history")
        assert endpoint_type == "historical"

    @pytest.mark.asyncio
    async def test_check_limit_exceeded(self, rate_limiter):
        """Test rate limit check when limit is exceeded."""
        # Add more than 60 requests in the last 2 seconds
        now = datetime.now(timezone.utc)
        for _ in range(61):
            rate_limiter.usage["rest"].append(now)

        result = await rate_limiter.check_limit("/api/v5/market/ticker", 1)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_limit_exception_handling(self, rate_limiter):
        """Test check_limit with exception handling."""
        # Mock _get_endpoint_type to raise an exception
        rate_limiter._get_endpoint_type = Mock(side_effect=Exception("Test error"))

        with pytest.raises(ExchangeRateLimitError):
            await rate_limiter.check_limit("/api/v5/market/ticker", 1)

    @pytest.mark.asyncio
    async def test_wait_for_reset_exception_handling(self, rate_limiter):
        """Test wait_for_reset with exception handling."""
        # Mock to raise an exception
        rate_limiter.usage = None  # This will cause an exception

        with pytest.raises(ExchangeRateLimitError):
            await rate_limiter.wait_for_reset("/api/v5/market/ticker")

    def test_get_endpoint_type_unknown(self, rate_limiter):
        """Test _get_endpoint_type with unknown endpoint."""
        result = rate_limiter._get_endpoint_type("/api/v5/unknown/endpoint")
        assert result == "rest"  # Should default to rest


class TestCoinbaseRateLimiter:
    """Test cases for CoinbaseRateLimiter class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Mock(spec=Config)

    @pytest.fixture
    def rate_limiter(self, config):
        """Create a CoinbaseRateLimiter instance."""
        return CoinbaseRateLimiter(config)

    def test_initialization(self, rate_limiter):
        """Test that the rate limiter initializes correctly."""
        assert rate_limiter.points_limit == 8000
        assert rate_limiter.private_limit == 10
        assert rate_limiter.public_limit == 15
        assert hasattr(rate_limiter, "points_usage")
        assert hasattr(rate_limiter, "private_usage")
        assert hasattr(rate_limiter, "public_usage")

    @pytest.mark.asyncio
    async def test_check_limit_valid_request(self, rate_limiter):
        """Test rate limit check with valid request."""
        result = await rate_limiter.check_limit("/products/BTC-USD/ticker", False)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_limit_invalid_endpoint(self, rate_limiter):
        """Test rate limit check with invalid endpoint."""
        with pytest.raises(ValidationError, match="Endpoint is required"):
            await rate_limiter.check_limit("", False)

    @pytest.mark.asyncio
    async def test_wait_for_reset_valid_request(self, rate_limiter):
        """Test wait for reset with valid request."""
        result = await rate_limiter.wait_for_reset("/products/BTC-USD/ticker")
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_wait_for_reset_invalid_endpoint(self, rate_limiter):
        """Test wait for reset with invalid endpoint."""
        with pytest.raises(ValidationError, match="Endpoint is required"):
            await rate_limiter.wait_for_reset("")

    def test_calculate_points_order(self, rate_limiter):
        """Test point calculation for order endpoints."""
        points = rate_limiter._calculate_points("/orders")
        assert points == 10

    def test_calculate_points_balance(self, rate_limiter):
        """Test point calculation for balance endpoints."""
        points = rate_limiter._calculate_points("/accounts/balance")
        assert points == 5

    def test_calculate_points_market(self, rate_limiter):
        """Test point calculation for market data endpoints."""
        points = rate_limiter._calculate_points("/products/BTC-USD/ticker")
        assert points == 1

    def test_calculate_points_default(self, rate_limiter):
        """Test point calculation for default endpoints."""
        points = rate_limiter._calculate_points("/some/other/endpoint")
        assert points == 2

    @pytest.mark.asyncio
    async def test_check_private_limit_valid(self, rate_limiter):
        """Test private limit check with valid state."""
        result = await rate_limiter._check_private_limit()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_private_limit_exceeded(self, rate_limiter):
        """Test private limit check when limit is exceeded."""
        # Add more than 10 requests in the last second
        now = datetime.now(timezone.utc)
        for _ in range(11):
            rate_limiter.private_usage.append(now)

        result = await rate_limiter._check_private_limit()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_public_limit_valid(self, rate_limiter):
        """Test public limit check with valid state."""
        result = await rate_limiter._check_public_limit()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_public_limit_exceeded(self, rate_limiter):
        """Test public limit check when limit is exceeded."""
        # Add more than 15 requests in the last second
        now = datetime.now(timezone.utc)
        for _ in range(16):
            rate_limiter.public_usage.append(now)

        result = await rate_limiter._check_public_limit()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_limit_exception_handling(self, rate_limiter):
        """Test check_limit with exception handling."""
        # Mock _check_private_limit to raise an exception
        rate_limiter._check_private_limit = AsyncMock(side_effect=Exception("Test error"))

        with pytest.raises(ExchangeRateLimitError):
            await rate_limiter.check_limit("/accounts", True)

    @pytest.mark.asyncio
    async def test_wait_for_reset_exception_handling(self, rate_limiter):
        """Test wait_for_reset with exception handling."""
        # Mock to raise an exception
        rate_limiter.points_usage = None  # This will cause an exception

        with pytest.raises(ExchangeRateLimitError):
            await rate_limiter.wait_for_reset("/products/BTC-USD/ticker")

    @pytest.mark.asyncio
    async def test_check_private_limit_exception_handling(self, rate_limiter):
        """Test _check_private_limit with exception handling."""
        # Mock to raise an exception
        rate_limiter.private_usage = None  # This will cause an exception

        with pytest.raises(Exception):
            await rate_limiter._check_private_limit()

    @pytest.mark.asyncio
    async def test_check_public_limit_exception_handling(self, rate_limiter):
        """Test _check_public_limit with exception handling."""
        # Mock to raise an exception
        rate_limiter.public_usage = None  # This will cause an exception

        with pytest.raises(Exception):
            await rate_limiter._check_public_limit()


class TestExchangeType:
    """Test cases for ExchangeType enum."""

    def test_exchange_types(self):
        """Test that all expected exchange types are defined."""
        assert ExchangeType.BINANCE.value == "binance"
        assert ExchangeType.OKX.value == "okx"
        assert ExchangeType.COINBASE.value == "coinbase"


class TestRequestType:
    """Test cases for RequestType enum."""

    def test_request_types(self):
        """Test that all expected request types are defined."""
        assert RequestType.MARKET_DATA.value == "market_data"
        assert RequestType.ORDER_PLACEMENT.value == "order_placement"
        assert RequestType.ORDER_CANCELLATION.value == "order_cancellation"
        assert RequestType.BALANCE_QUERY.value == "balance_query"
        assert RequestType.POSITION_QUERY.value == "position_query"
        assert RequestType.HISTORICAL_DATA.value == "historical_data"
        assert RequestType.WEBSOCKET_CONNECTION.value == "websocket_connection"


# Integration tests
class TestAdvancedRateLimiterIntegration:
    """Integration tests for advanced rate limiting."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Mock(spec=Config)

    @pytest.mark.asyncio
    async def test_multi_exchange_rate_limiting(self, config):
        """Test rate limiting across multiple exchanges."""
        rate_limiter = AdvancedRateLimiter(config)

        # Test rate limiting for different exchanges
        exchanges = ["binance", "okx", "coinbase"]
        endpoints = ["/api/v3/ticker/price", "/api/v5/market/ticker", "/products/BTC-USD/ticker"]

        for exchange, endpoint in zip(exchanges, endpoints, strict=False):
            result = await rate_limiter.check_rate_limit(exchange, endpoint, 1)
            assert result is True

    @pytest.mark.asyncio
    async def test_rate_limit_coordination(self, config):
        """Test coordination between different rate limiters."""
        rate_limiter = AdvancedRateLimiter(config)

        # Test that each exchange limiter is created properly via lazy initialization
        binance_limiter = rate_limiter._get_or_create_limiter("binance")
        okx_limiter = rate_limiter._get_or_create_limiter("okx")
        coinbase_limiter = rate_limiter._get_or_create_limiter("coinbase")
        
        assert isinstance(binance_limiter, BinanceRateLimiter)
        assert isinstance(okx_limiter, OKXRateLimiter)
        assert isinstance(coinbase_limiter, CoinbaseRateLimiter)

    @pytest.mark.asyncio
    async def test_error_handling(self, config):
        """Test error handling in rate limiting."""
        rate_limiter = AdvancedRateLimiter(config)

        # Test with invalid exchange
        with pytest.raises(ExchangeRateLimitError):
            await rate_limiter.check_rate_limit("invalid", "/api/test", 1)

        # Test with invalid parameters
        with pytest.raises(ValidationError):
            await rate_limiter.check_rate_limit("binance", "", 1)

        with pytest.raises(ValidationError):
            await rate_limiter.check_rate_limit("binance", "/api/test", 0)
