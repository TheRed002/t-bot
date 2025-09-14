"""Unit tests for backtesting controller."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict

from src.backtesting.controller import BacktestController
from src.core.exceptions import ServiceError, ValidationError


class TestBacktestController:
    """Test cases for BacktestController."""

    @pytest.fixture
    def mock_service(self):
        """Mock backtest service."""
        service = AsyncMock()
        service.run_backtest_from_dict.return_value = {"id": "test", "status": "completed"}
        service.get_active_backtests.return_value = {"test": {"status": "active"}}
        service.cancel_backtest.return_value = True
        service.clear_cache.return_value = 5
        service.get_cache_stats.return_value = {"keys": 10, "hits": 100}
        service.health_check.return_value = MagicMock(status="healthy")
        return service

    @pytest.fixture
    def controller(self, mock_service):
        """Create controller instance."""
        return BacktestController(mock_service)

    @pytest.mark.asyncio
    async def test_run_backtest_success(self, controller, mock_service):
        """Test successful backtest run."""
        request_data = {"strategy": "test", "symbols": ["BTCUSD"]}
        mock_result = {"id": "test", "status": "completed"}
        mock_service.run_backtest_from_dict.return_value = mock_result
        mock_service.serialize_result.return_value = {"serialized": "result"}

        result = await controller.run_backtest(request_data)

        assert result == {"serialized": "result"}
        mock_service.run_backtest_from_dict.assert_called_once_with(request_data)
        mock_service.serialize_result.assert_called_once_with(mock_result)

    @pytest.mark.asyncio
    async def test_run_backtest_validation_error(self, controller, mock_service):
        """Test backtest run with validation error."""
        mock_service.run_backtest_from_dict.side_effect = ValidationError("Invalid data")

        with pytest.raises(ValidationError, match="Invalid data"):
            await controller.run_backtest({})

    @pytest.mark.asyncio
    async def test_run_backtest_service_error(self, controller, mock_service):
        """Test backtest run with service error."""
        mock_service.run_backtest_from_dict.side_effect = ServiceError("Service failed")

        with pytest.raises(ServiceError, match="Service failed"):
            await controller.run_backtest({})

    @pytest.mark.asyncio
    async def test_get_active_backtests_success(self, controller, mock_service):
        """Test getting active backtests."""
        mock_service.get_active_backtests.return_value = {"test": {"status": "active"}}

        result = await controller.get_active_backtests()

        assert result == {"test": {"status": "active"}}
        mock_service.get_active_backtests.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_active_backtests_error(self, controller, mock_service):
        """Test getting active backtests with error."""
        mock_service.get_active_backtests.side_effect = Exception("Failed")

        with pytest.raises(ServiceError, match="Failed to get active backtests"):
            await controller.get_active_backtests()

    @pytest.mark.asyncio
    async def test_cancel_backtest_success(self, controller, mock_service):
        """Test successful backtest cancellation."""
        mock_service.cancel_backtest.return_value = True

        result = await controller.cancel_backtest("test_id")

        assert result == {"cancelled": True, "backtest_id": "test_id"}
        mock_service.cancel_backtest.assert_called_once_with("test_id")

    @pytest.mark.asyncio
    async def test_cancel_backtest_not_found(self, controller, mock_service):
        """Test backtest cancellation when not found."""
        mock_service.cancel_backtest.return_value = False

        result = await controller.cancel_backtest("missing_id")

        assert result == {"cancelled": False, "backtest_id": "missing_id"}

    @pytest.mark.asyncio
    async def test_cancel_backtest_error(self, controller, mock_service):
        """Test backtest cancellation with error."""
        mock_service.cancel_backtest.side_effect = Exception("Failed")

        with pytest.raises(ServiceError, match="Failed to cancel backtest"):
            await controller.cancel_backtest("test_id")

    @pytest.mark.asyncio
    async def test_clear_cache_success(self, controller, mock_service):
        """Test successful cache clearing."""
        mock_service.clear_cache.return_value = 5

        result = await controller.clear_cache("test*")

        assert result == {"cleared_entries": 5, "pattern": "test*"}
        mock_service.clear_cache.assert_called_once_with("test*")

    @pytest.mark.asyncio
    async def test_clear_cache_default_pattern(self, controller, mock_service):
        """Test cache clearing with default pattern."""
        await controller.clear_cache()

        mock_service.clear_cache.assert_called_once_with("*")

    @pytest.mark.asyncio
    async def test_clear_cache_error(self, controller, mock_service):
        """Test cache clearing with error."""
        mock_service.clear_cache.side_effect = Exception("Failed")

        with pytest.raises(ServiceError, match="Failed to clear cache"):
            await controller.clear_cache()

    @pytest.mark.asyncio
    async def test_get_cache_stats_success(self, controller, mock_service):
        """Test getting cache stats."""
        mock_service.get_cache_stats.return_value = {"keys": 10, "hits": 100}

        result = await controller.get_cache_stats()

        assert result == {"keys": 10, "hits": 100}
        mock_service.get_cache_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache_stats_error(self, controller, mock_service):
        """Test getting cache stats with error."""
        mock_service.get_cache_stats.side_effect = Exception("Failed")

        with pytest.raises(ServiceError, match="Failed to get cache stats"):
            await controller.get_cache_stats()

    @pytest.mark.asyncio
    async def test_health_check_success(self, controller, mock_service):
        """Test successful health check."""
        from src.core.base.interfaces import HealthCheckResult, HealthStatus
        mock_health = HealthCheckResult(status=HealthStatus.HEALTHY, details={})
        mock_service.health_check.return_value = mock_health

        result = await controller.health_check()

        assert result.status == HealthStatus.HEALTHY
        assert "component" in result.details
        mock_service.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_error(self, controller, mock_service):
        """Test health check with error."""
        from src.core.base.interfaces import HealthStatus
        mock_service.health_check.side_effect = Exception("Failed")

        result = await controller.health_check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "error" in result.details

    @pytest.mark.asyncio
    async def test_cleanup_success(self, controller, mock_service):
        """Test successful cleanup."""
        await controller.cleanup()

        mock_service.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_error(self, controller, mock_service):
        """Test cleanup with error - should not raise."""
        mock_service.cleanup.side_effect = Exception("Cleanup failed")

        # Should not raise exception
        await controller.cleanup()

        mock_service.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_backtest_empty_id(self, controller, mock_service):
        """Test cancelling backtest with empty ID."""
        with pytest.raises(ValidationError, match="Backtest ID is required"):
            await controller.cancel_backtest("")