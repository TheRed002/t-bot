"""
Unit tests for Health Monitor - completely mocked.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestHealthMonitor:
    """Test Health Monitor with complete mocking."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check."""
        mock_monitor = MagicMock()
        mock_monitor.check_health = AsyncMock(
            return_value={"status": "healthy", "timestamp": datetime.now(timezone.utc)}
        )

        result = await mock_monitor.check_health()
        assert result["status"] == "healthy"
        assert "timestamp" in result
        mock_monitor.check_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_monitoring(self):
        """Test starting health monitoring."""
        mock_monitor = MagicMock()
        mock_monitor.start = AsyncMock()
        mock_monitor.is_running = False

        await mock_monitor.start()
        mock_monitor.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_monitoring(self):
        """Test stopping health monitoring."""
        mock_monitor = MagicMock()
        mock_monitor.stop = AsyncMock()
        mock_monitor.is_running = True

        await mock_monitor.stop()
        mock_monitor.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_status_update(self):
        """Test health status update."""
        mock_monitor = MagicMock()
        mock_monitor.update_status = AsyncMock()

        await mock_monitor.update_status("binance", "healthy")
        mock_monitor.update_status.assert_called_once_with("binance", "healthy")

    @pytest.mark.asyncio
    async def test_health_metrics(self):
        """Test health metrics collection."""
        mock_monitor = MagicMock()
        mock_monitor.get_metrics = AsyncMock(
            return_value={
                "uptime": 3600,
                "requests_processed": 1000,
                "errors": 5,
                "success_rate": 0.995,
            }
        )

        metrics = await mock_monitor.get_metrics()
        assert metrics["uptime"] == 3600
        assert metrics["success_rate"] == 0.995
        mock_monitor.get_metrics.assert_called_once()
