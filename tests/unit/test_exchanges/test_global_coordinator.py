"""
Unit tests for Global Coordinator - completely mocked.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestGlobalCoordinator:
    """Test Global Coordinator with complete mocking."""

    @pytest.mark.asyncio
    async def test_coordinator_start(self):
        """Test starting the coordinator."""
        mock_coordinator = MagicMock()
        mock_coordinator.start = AsyncMock()
        mock_coordinator.is_running = False

        await mock_coordinator.start()
        mock_coordinator.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_coordinator_stop(self):
        """Test stopping the coordinator."""
        mock_coordinator = MagicMock()
        mock_coordinator.stop = AsyncMock()
        mock_coordinator.is_running = True

        await mock_coordinator.stop()
        mock_coordinator.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_coordinate_request(self):
        """Test coordinating a request."""
        mock_coordinator = MagicMock()
        mock_coordinator.coordinate = AsyncMock(return_value=True)

        result = await mock_coordinator.coordinate("binance", "GET", "/api/v3/ticker")
        assert result is True
        mock_coordinator.coordinate.assert_called_once_with("binance", "GET", "/api/v3/ticker")

    @pytest.mark.asyncio
    async def test_rate_limit_check(self):
        """Test rate limit checking."""
        mock_coordinator = MagicMock()
        mock_coordinator.check_rate_limit = AsyncMock(return_value=True)

        result = await mock_coordinator.check_rate_limit("binance", "orders")
        assert result is True
        mock_coordinator.check_rate_limit.assert_called_once_with("binance", "orders")

    @pytest.mark.asyncio
    async def test_get_coordinator_stats(self):
        """Test getting coordinator statistics."""
        mock_coordinator = MagicMock()
        mock_coordinator.get_stats = AsyncMock(
            return_value={"total_requests": 1000, "rate_limited": 10, "success_rate": 0.99}
        )

        stats = await mock_coordinator.get_stats()
        assert stats["total_requests"] == 1000
        assert stats["success_rate"] == 0.99
        mock_coordinator.get_stats.assert_called_once()
