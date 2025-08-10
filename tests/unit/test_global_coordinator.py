"""
Unit tests for global rate coordinator.

This module tests the global rate coordination functionality including
cross-exchange rate limit management and global limits.

CRITICAL: Tests must achieve 90% coverage for P-007 components.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.types import RequestType

# Import the components to test
from src.exchanges.global_coordinator import GlobalRateCoordinator


class TestGlobalRateCoordinator:
    """Test cases for GlobalRateCoordinator class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Mock(spec=Config)

    @pytest.fixture
    def coordinator(self, config):
        """Create a GlobalRateCoordinator instance."""
        return GlobalRateCoordinator(config)

    def test_initialization(self, coordinator):
        """Test that the coordinator initializes correctly."""
        assert coordinator is not None
        assert hasattr(coordinator, "global_limits")
        assert hasattr(coordinator, "request_history")
        assert hasattr(coordinator, "order_history")
        assert hasattr(coordinator, "connection_history")
        assert hasattr(coordinator, "exchange_usage")

        # Check global limits
        expected_limits = {
            "total_requests_per_minute": 5000,
            "orders_per_minute": 1000,
            "concurrent_connections": 50,
            "websocket_messages_per_second": 1000,
        }
        assert coordinator.global_limits == expected_limits

    @pytest.mark.asyncio
    async def test_check_global_limits_valid_request(self, coordinator):
        """Test global limits check with valid request."""
        result = await coordinator.check_global_limits(RequestType.MARKET_DATA.value, 1)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_global_limits_invalid_request_type(self, coordinator):
        """Test global limits check with invalid request type."""
        with pytest.raises(ValidationError, match="Invalid request type"):
            await coordinator.check_global_limits("invalid_type", 1)

    @pytest.mark.asyncio
    async def test_check_global_limits_invalid_count(self, coordinator):
        """Test global limits check with invalid count."""
        with pytest.raises(ValidationError, match="Count must be positive"):
            await coordinator.check_global_limits(RequestType.MARKET_DATA.value, 0)

    @pytest.mark.asyncio
    async def test_check_global_limits_missing_request_type(self, coordinator):
        """Test global limits check with missing request type."""
        with pytest.raises(ValidationError, match="Request type is required"):
            await coordinator.check_global_limits("", 1)

    @pytest.mark.asyncio
    async def test_coordinate_request_valid(self, coordinator):
        """Test request coordination with valid parameters."""
        result = await coordinator.coordinate_request(
            "binance", "/api/v3/ticker/price", RequestType.MARKET_DATA.value
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_coordinate_request_invalid_exchange(self, coordinator):
        """Test request coordination with invalid exchange."""
        with pytest.raises(
            ValidationError, match="Exchange, endpoint, and request_type are required"
        ):
            await coordinator.coordinate_request(
                "", "/api/v3/ticker/price", RequestType.MARKET_DATA.value
            )

    @pytest.mark.asyncio
    async def test_coordinate_request_invalid_endpoint(self, coordinator):
        """Test request coordination with invalid endpoint."""
        with pytest.raises(
            ValidationError, match="Exchange, endpoint, and request_type are required"
        ):
            await coordinator.coordinate_request("binance", "", RequestType.MARKET_DATA.value)

    @pytest.mark.asyncio
    async def test_coordinate_request_invalid_request_type(self, coordinator):
        """Test request coordination with invalid request type."""
        with pytest.raises(
            ValidationError, match="Exchange, endpoint, and request_type are required"
        ):
            await coordinator.coordinate_request("binance", "/api/v3/ticker/price", "")

    @pytest.mark.asyncio
    async def test_check_order_limits_valid(self, coordinator):
        """Test order limits check with valid state."""
        result = await coordinator._check_order_limits(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_order_limits_exceeded(self, coordinator):
        """Test order limits check when limit is exceeded."""
        # Add more than 1000 orders in the last minute
        now = datetime.now()
        for _ in range(1001):
            coordinator.order_history.append(now)

        result = await coordinator._check_order_limits(1)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_connection_limits_valid(self, coordinator):
        """Test connection limits check with valid state."""
        result = await coordinator._check_connection_limits(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_connection_limits_exceeded(self, coordinator):
        """Test connection limits check when limit is exceeded."""
        # Add more than 50 connections in the last minute
        now = datetime.now()
        for _ in range(51):
            coordinator.connection_history.append(now)

        result = await coordinator._check_connection_limits(1)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_general_request_limits_valid(self, coordinator):
        """Test general request limits check with valid state."""
        result = await coordinator._check_general_request_limits(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_general_request_limits_exceeded(self, coordinator):
        """Test general request limits check when limit is exceeded."""
        # Add more than 5000 requests in the last minute
        now = datetime.now()
        for _ in range(5001):
            coordinator.request_history["general"].append(now)

        result = await coordinator._check_general_request_limits(1)
        assert result is False

    def test_record_request_order_placement(self, coordinator):
        """Test recording order placement request."""
        coordinator._record_request(RequestType.ORDER_PLACEMENT.value, 5)
        assert len(coordinator.order_history) == 5

    def test_record_request_websocket_connection(self, coordinator):
        """Test recording websocket connection request."""
        coordinator._record_request(RequestType.WEBSOCKET_CONNECTION.value, 3)
        assert len(coordinator.connection_history) == 3

    def test_record_request_general(self, coordinator):
        """Test recording general request."""
        coordinator._record_request(RequestType.MARKET_DATA.value, 2)
        assert len(coordinator.request_history["general"]) == 2

    def test_record_request_cleanup(self, coordinator):
        """Test that old request history is cleaned up."""
        # Add more than 10000 requests
        for _ in range(10001):
            coordinator._record_request(RequestType.MARKET_DATA.value, 1)

        assert len(coordinator.request_history["general"]) == 10000

    def test_record_exchange_usage(self, coordinator):
        """Test recording exchange usage."""
        exchange = "binance"
        endpoint = "/api/v3/ticker/price"
        request_type = RequestType.MARKET_DATA.value

        coordinator._record_exchange_usage(exchange, endpoint, request_type)

        assert exchange in coordinator.exchange_usage
        assert endpoint in coordinator.exchange_usage[exchange]
        assert len(coordinator.exchange_usage[exchange][endpoint]) == 1

    def test_record_exchange_usage_cleanup(self, coordinator):
        """Test that old exchange usage is cleaned up."""
        exchange = "binance"
        endpoint = "/api/v3/ticker/price"
        request_type = RequestType.MARKET_DATA.value

        # Add more than 1000 requests
        for _ in range(1001):
            coordinator._record_exchange_usage(exchange, endpoint, request_type)

        assert len(coordinator.exchange_usage[exchange][endpoint]) == 1000

    def test_get_global_usage_stats(self, coordinator):
        """Test getting global usage statistics."""
        # Add some test data
        now = datetime.now()
        coordinator.order_history.append(now)
        coordinator.connection_history.append(now)
        coordinator.request_history["general"].append(now)

        stats = coordinator.get_global_usage_stats()

        assert "orders_per_minute" in stats
        assert "concurrent_connections" in stats
        assert "total_requests_per_minute" in stats
        assert "limits" in stats
        assert "timestamp" in stats

        assert stats["orders_per_minute"] == 1
        assert stats["concurrent_connections"] == 1
        assert stats["total_requests_per_minute"] == 1

    def test_get_exchange_usage_stats_empty(self, coordinator):
        """Test getting exchange usage stats for non-existent exchange."""
        stats = coordinator.get_exchange_usage_stats("nonexistent")
        assert stats == {}

    def test_get_exchange_usage_stats_with_data(self, coordinator):
        """Test getting exchange usage stats with data."""
        exchange = "binance"
        endpoint = "/api/v3/ticker/price"
        request_type = RequestType.MARKET_DATA.value

        # Add some test data
        coordinator._record_exchange_usage(exchange, endpoint, request_type)

        stats = coordinator.get_exchange_usage_stats(exchange)

        assert endpoint in stats
        assert "requests_per_minute" in stats[endpoint]
        assert "total_requests" in stats[endpoint]
        assert stats[endpoint]["requests_per_minute"] == 1
        assert stats[endpoint]["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_wait_for_global_capacity_order_placement(self, coordinator):
        """Test waiting for order placement capacity."""
        result = await coordinator.wait_for_global_capacity(RequestType.ORDER_PLACEMENT.value, 1)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_wait_for_global_capacity_websocket_connection(self, coordinator):
        """Test waiting for websocket connection capacity."""
        result = await coordinator.wait_for_global_capacity(
            RequestType.WEBSOCKET_CONNECTION.value, 1
        )
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_wait_for_global_capacity_general(self, coordinator):
        """Test waiting for general request capacity."""
        result = await coordinator.wait_for_global_capacity(RequestType.MARKET_DATA.value, 1)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_wait_for_global_capacity_invalid_request_type(self, coordinator):
        """Test waiting for capacity with invalid request type."""
        with pytest.raises(ValidationError, match="Request type is required"):
            await coordinator.wait_for_global_capacity("", 1)

    @pytest.mark.asyncio
    async def test_wait_for_global_capacity_invalid_count(self, coordinator):
        """Test waiting for capacity with invalid count."""
        with pytest.raises(ValidationError, match="Count must be positive"):
            await coordinator.wait_for_global_capacity(RequestType.MARKET_DATA.value, 0)

    @pytest.mark.asyncio
    async def test_calculate_order_wait_time_no_wait(self, coordinator):
        """Test calculating order wait time when no wait is needed."""
        result = await coordinator._calculate_order_wait_time(1)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_connection_wait_time_no_wait(self, coordinator):
        """Test calculating connection wait time when no wait is needed."""
        result = await coordinator._calculate_connection_wait_time(1)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_calculate_general_wait_time_no_wait(self, coordinator):
        """Test calculating general wait time when no wait is needed."""
        result = await coordinator._calculate_general_wait_time(1)
        assert result == 0.0

    def test_reset_global_limits(self, coordinator):
        """Test resetting global limits."""
        # Add some test data
        now = datetime.now()
        coordinator.order_history.append(now)
        coordinator.connection_history.append(now)
        coordinator.request_history["general"].append(now)
        coordinator.exchange_usage["binance"] = {"test": [now]}

        # Reset
        coordinator.reset_global_limits()

        # Verify everything is cleared
        assert len(coordinator.order_history) == 0
        assert len(coordinator.connection_history) == 0
        assert len(coordinator.request_history["general"]) == 0
        assert len(coordinator.exchange_usage) == 0

    def test_update_global_limits(self, coordinator):
        """Test updating global limits."""
        new_limits = {
            "total_requests_per_minute": 6000,
            "orders_per_minute": 1200,
        }

        coordinator.update_global_limits(new_limits)

        assert coordinator.global_limits["total_requests_per_minute"] == 6000
        assert coordinator.global_limits["orders_per_minute"] == 1200

    def test_update_global_limits_unknown_key(self, coordinator):
        """Test updating global limits with unknown key."""
        new_limits = {
            "unknown_limit": 1000,
        }

        # Should not raise an exception, just log a warning
        coordinator.update_global_limits(new_limits)

    def test_get_health_status(self, coordinator):
        """Test getting health status."""
        health = coordinator.get_health_status()

        assert "status" in health
        assert "order_usage_percent" in health
        assert "connection_usage_percent" in health
        assert "request_usage_percent" in health
        assert "timestamp" in health

        # All percentages should be 0 initially
        assert health["order_usage_percent"] == 0.0
        assert health["connection_usage_percent"] == 0.0
        assert health["request_usage_percent"] == 0.0
        assert health["status"] == "healthy"


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
class TestGlobalRateCoordinatorIntegration:
    """Integration tests for global rate coordination."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Mock(spec=Config)

    @pytest.mark.asyncio
    async def test_multi_request_type_coordination(self, config):
        """Test coordination across different request types."""
        coordinator = GlobalRateCoordinator(config)

        # Test different request types
        request_types = [
            RequestType.MARKET_DATA.value,
            RequestType.ORDER_PLACEMENT.value,
            RequestType.WEBSOCKET_CONNECTION.value,
        ]

        for request_type in request_types:
            result = await coordinator.check_global_limits(request_type, 1)
            assert result is True

    @pytest.mark.asyncio
    async def test_exchange_coordination(self, config):
        """Test coordination across different exchanges."""
        coordinator = GlobalRateCoordinator(config)

        exchanges = ["binance", "okx", "coinbase"]
        endpoints = ["/api/v3/ticker/price", "/api/v5/market/ticker", "/products/BTC-USD/ticker"]
        request_types = [RequestType.MARKET_DATA.value] * 3

        for exchange, endpoint, request_type in zip(
            exchanges, endpoints, request_types, strict=False
        ):
            result = await coordinator.coordinate_request(exchange, endpoint, request_type)
            assert result is True

    @pytest.mark.asyncio
    async def test_limit_exceeded_scenarios(self, config):
        """Test scenarios where limits are exceeded."""
        coordinator = GlobalRateCoordinator(config)

        # Test order limit exceeded
        now = datetime.now()
        for _ in range(1001):
            coordinator.order_history.append(now)

        result = await coordinator.check_global_limits(RequestType.ORDER_PLACEMENT.value, 1)
        assert result is False

        # Reset and test connection limit exceeded
        coordinator.reset_global_limits()
        for _ in range(51):
            coordinator.connection_history.append(now)

        result = await coordinator.check_global_limits(RequestType.WEBSOCKET_CONNECTION.value, 1)
        assert result is False

    @pytest.mark.asyncio
    async def test_error_handling(self, config):
        """Test error handling in global coordination."""
        coordinator = GlobalRateCoordinator(config)

        # Test with invalid request type
        with pytest.raises(ValidationError):
            await coordinator.check_global_limits("invalid_type", 1)

        # Test with invalid count
        with pytest.raises(ValidationError):
            await coordinator.check_global_limits(RequestType.MARKET_DATA.value, 0)

        # Test with missing parameters
        with pytest.raises(ValidationError):
            await coordinator.coordinate_request("", "/api/test", RequestType.MARKET_DATA.value)

    @pytest.mark.asyncio
    async def test_usage_tracking(self, config):
        """Test usage tracking functionality."""
        coordinator = GlobalRateCoordinator(config)

        # Add some usage data
        coordinator._record_request(RequestType.ORDER_PLACEMENT.value, 5)
        coordinator._record_request(RequestType.WEBSOCKET_CONNECTION.value, 3)
        coordinator._record_request(RequestType.MARKET_DATA.value, 10)

        # Check usage stats
        stats = coordinator.get_global_usage_stats()
        assert stats["orders_per_minute"] == 5
        assert stats["concurrent_connections"] == 3
        assert stats["total_requests_per_minute"] == 10

    @pytest.mark.asyncio
    async def test_health_monitoring(self, config):
        """Test health monitoring functionality."""
        coordinator = GlobalRateCoordinator(config)

        # Get initial health status
        health = coordinator.get_health_status()
        assert health["status"] == "healthy"

        # Add usage to trigger warning status
        now = datetime.now()
        for _ in range(4500):  # 90% of 5000 limit
            coordinator.request_history["general"].append(now)

        health = coordinator.get_health_status()
        assert health["request_usage_percent"] == 90.0
        assert health["status"] == "warning"
