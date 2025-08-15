"""Unit tests for BotMonitor component."""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import Config
from src.core.types import BotStatus, BotMetrics, BotPriority
from src.bot_management.bot_monitor import BotMonitor


@pytest.fixture
def config():
    """Create test configuration."""
    config = MagicMock(spec=Config)
    config.error_handling = MagicMock()
    config.bot_management = {
        "health_check_interval": 30,
        "performance_window_minutes": 60,
        "alert_thresholds": {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "error_rate": 0.1
        }
    }
    config.monitoring = {
        "influxdb_enabled": False,
        "alerts_enabled": True
    }
    return config


@pytest.fixture
def bot_metrics():
    """Create test bot metrics."""
    return BotMetrics(
        cpu_usage=45.0,
        memory_usage=60.0,
        active_positions=3,
        orders_per_minute=5.2,
        error_rate=0.02,
        latency_ms=125.0,
        uptime_seconds=3600,
        total_trades=25,
        successful_trades=23,
        pnl=Decimal("150.50"),
        timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def monitor(config):
    """Create BotMonitor for testing."""
    return BotMonitor(config)


class TestBotMonitor:
    """Test cases for BotMonitor class."""

    @pytest.mark.asyncio
    async def test_monitor_initialization(self, monitor, config):
        """Test monitor initialization."""
        assert monitor.config == config
        assert monitor.bot_metrics == {}
        assert monitor.alert_history == []
        assert monitor.performance_baselines == {}
        assert not monitor.is_running

    @pytest.mark.asyncio
    async def test_start_monitor(self, monitor):
        """Test monitor startup."""
        await monitor.start()
        
        assert monitor.is_running
        assert monitor.monitoring_task is not None

    @pytest.mark.asyncio
    async def test_stop_monitor(self, monitor):
        """Test monitor shutdown."""
        await monitor.start()
        await monitor.stop()
        
        assert not monitor.is_running

    @pytest.mark.asyncio
    async def test_update_bot_metrics(self, monitor, bot_metrics):
        """Test bot metrics update."""
        bot_id = "test_bot_001"
        
        await monitor.update_bot_metrics(bot_id, bot_metrics)
        
        assert bot_id in monitor.bot_metrics
        assert monitor.bot_metrics[bot_id] == bot_metrics
        assert monitor.monitoring_statistics["total_metrics_updates"] == 1

    @pytest.mark.asyncio
    async def test_get_bot_metrics(self, monitor, bot_metrics):
        """Test retrieving bot metrics."""
        bot_id = "test_bot_001"
        
        # Add metrics
        await monitor.update_bot_metrics(bot_id, bot_metrics)
        
        # Retrieve metrics
        retrieved = await monitor.get_bot_metrics(bot_id)
        
        assert retrieved == bot_metrics

    @pytest.mark.asyncio
    async def test_get_bot_metrics_not_found(self, monitor):
        """Test retrieving metrics for non-existent bot."""
        retrieved = await monitor.get_bot_metrics("non_existent")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_check_bot_health_healthy(self, monitor, bot_metrics):
        """Test health check for healthy bot."""
        bot_id = "test_bot_001"
        bot_status = BotStatus.RUNNING
        
        # Update with healthy metrics
        await monitor.update_bot_metrics(bot_id, bot_metrics)
        
        health_report = await monitor.check_bot_health(bot_id, bot_status)
        
        assert health_report["overall_health"] == "healthy"
        assert health_report["health_score"] > 0.7
        assert len(health_report["issues"]) == 0

    @pytest.mark.asyncio
    async def test_check_bot_health_unhealthy(self, monitor, bot_metrics):
        """Test health check for unhealthy bot."""
        bot_id = "test_bot_001"
        bot_status = BotStatus.RUNNING
        
        # Create unhealthy metrics
        unhealthy_metrics = BotMetrics(
            cpu_usage=95.0,  # High CPU
            memory_usage=90.0,  # High memory
            active_positions=3,
            orders_per_minute=0.1,  # Low activity
            error_rate=0.15,  # High error rate
            latency_ms=1500.0,  # High latency
            uptime_seconds=3600,
            total_trades=10,
            successful_trades=7,  # Low success rate
            pnl=Decimal("-50.00"),  # Negative PnL
            timestamp=datetime.now(timezone.utc)
        )
        
        await monitor.update_bot_metrics(bot_id, unhealthy_metrics)
        
        health_report = await monitor.check_bot_health(bot_id, bot_status)
        
        assert health_report["overall_health"] in ["warning", "critical"]
        assert health_report["health_score"] < 0.5
        assert len(health_report["issues"]) > 0

    @pytest.mark.asyncio
    async def test_check_bot_health_no_metrics(self, monitor):
        """Test health check for bot with no metrics."""
        bot_id = "test_bot_001"
        bot_status = BotStatus.RUNNING
        
        health_report = await monitor.check_bot_health(bot_id, bot_status)
        
        assert health_report["overall_health"] == "unknown"
        assert "No metrics available" in health_report["issues"]

    @pytest.mark.asyncio
    async def test_generate_alert_threshold_exceeded(self, monitor, bot_metrics):
        """Test alert generation for threshold exceeded."""
        bot_id = "test_bot_001"
        
        # Create metrics that exceed thresholds
        high_cpu_metrics = BotMetrics(
            cpu_usage=85.0,  # Exceeds 80% threshold
            memory_usage=60.0,
            active_positions=3,
            orders_per_minute=5.2,
            error_rate=0.02,
            latency_ms=125.0,
            uptime_seconds=3600,
            total_trades=25,
            successful_trades=23,
            pnl=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc)
        )
        
        await monitor.update_bot_metrics(bot_id, high_cpu_metrics)
        
        # Check for alerts
        await monitor._check_alert_conditions(bot_id)
        
        # Should have generated CPU alert
        cpu_alerts = [alert for alert in monitor.alert_history 
                     if alert["alert_type"] == "high_cpu_usage"]
        assert len(cpu_alerts) > 0

    @pytest.mark.asyncio
    async def test_performance_baseline_establishment(self, monitor, bot_metrics):
        """Test performance baseline establishment."""
        bot_id = "test_bot_001"
        
        # Add multiple metrics to establish baseline
        for i in range(10):
            metrics = BotMetrics(
                cpu_usage=40.0 + i,
                memory_usage=50.0 + i,
                active_positions=3,
                orders_per_minute=5.0,
                error_rate=0.02,
                latency_ms=100.0 + i * 5,
                uptime_seconds=3600 + i * 60,
                total_trades=25 + i,
                successful_trades=23 + i,
                pnl=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc)
            )
            await monitor.update_bot_metrics(bot_id, metrics)
        
        # Establish baseline
        await monitor._establish_performance_baseline(bot_id)
        
        assert bot_id in monitor.performance_baselines
        baseline = monitor.performance_baselines[bot_id]
        assert "cpu_usage" in baseline
        assert "memory_usage" in baseline
        assert baseline["cpu_usage"]["mean"] > 0

    @pytest.mark.asyncio
    async def test_detect_anomalies(self, monitor):
        """Test anomaly detection."""
        bot_id = "test_bot_001"
        
        # Establish baseline first
        for i in range(10):
            normal_metrics = BotMetrics(
                cpu_usage=45.0,
                memory_usage=60.0,
                active_positions=3,
                orders_per_minute=5.0,
                error_rate=0.02,
                latency_ms=125.0,
                uptime_seconds=3600,
                total_trades=25,
                successful_trades=23,
                pnl=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc)
            )
            await monitor.update_bot_metrics(bot_id, normal_metrics)
        
        await monitor._establish_performance_baseline(bot_id)
        
        # Add anomalous metrics
        anomalous_metrics = BotMetrics(
            cpu_usage=95.0,  # Significantly higher than baseline
            memory_usage=60.0,
            active_positions=3,
            orders_per_minute=0.5,  # Much lower than baseline
            error_rate=0.02,
            latency_ms=125.0,
            uptime_seconds=3600,
            total_trades=25,
            successful_trades=23,
            pnl=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc)
        )
        
        anomalies = await monitor._detect_anomalies(bot_id, anomalous_metrics)
        
        assert len(anomalies) > 0
        assert any("cpu_usage" in anomaly for anomaly in anomalies)

    @pytest.mark.asyncio
    async def test_get_monitoring_summary(self, monitor):
        """Test monitoring summary generation."""
        # Add metrics for multiple bots
        for i in range(3):
            bot_id = f"bot_{i}"
            metrics = BotMetrics(
                cpu_usage=45.0 + i * 10,
                memory_usage=60.0 + i * 5,
                active_positions=3,
                orders_per_minute=5.0,
                error_rate=0.02,
                latency_ms=125.0,
                uptime_seconds=3600,
                total_trades=25,
                successful_trades=23,
                pnl=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc)
            )
            await monitor.update_bot_metrics(bot_id, metrics)
        
        # Add some alerts
        monitor.alert_history.append({
            "alert_id": "test_alert",
            "bot_id": "bot_0",
            "alert_type": "high_cpu_usage",
            "severity": "warning",
            "timestamp": datetime.now(timezone.utc)
        })
        
        summary = await monitor.get_monitoring_summary()
        
        # Verify summary structure
        assert "monitoring_overview" in summary
        assert "bot_health_summary" in summary
        assert "alert_summary" in summary
        assert "performance_overview" in summary
        assert "system_health" in summary
        
        # Verify content
        assert summary["monitoring_overview"]["monitored_bots"] == 3
        assert summary["alert_summary"]["total_alerts"] == 1

    @pytest.mark.asyncio
    async def test_get_bot_health_history(self, monitor, bot_metrics):
        """Test bot health history retrieval."""
        bot_id = "test_bot_001"
        
        # Add multiple health records
        for i in range(5):
            await monitor.update_bot_metrics(bot_id, bot_metrics)
            await monitor.check_bot_health(bot_id, BotStatus.RUNNING)
        
        history = await monitor.get_bot_health_history(bot_id, hours=24)
        
        assert len(history) > 0
        assert all("timestamp" in record for record in history)
        assert all("health_score" in record for record in history)

    @pytest.mark.asyncio
    async def test_get_alert_history(self, monitor):
        """Test alert history retrieval."""
        # Add test alerts
        for i in range(5):
            alert = {
                "alert_id": f"alert_{i}",
                "bot_id": f"bot_{i % 2}",
                "alert_type": "test_alert",
                "severity": "warning",
                "timestamp": datetime.now(timezone.utc),
                "message": f"Test alert {i}"
            }
            monitor.alert_history.append(alert)
        
        # Get all alerts
        all_alerts = await monitor.get_alert_history()
        assert len(all_alerts) == 5
        
        # Get filtered alerts
        bot_alerts = await monitor.get_alert_history(bot_id="bot_0")
        assert len(bot_alerts) == 3  # bot_0 appears at indices 0, 2, 4

    @pytest.mark.asyncio
    async def test_calculate_health_score(self, monitor, bot_metrics):
        """Test health score calculation."""
        bot_id = "test_bot_001"
        bot_status = BotStatus.RUNNING
        
        score = await monitor._calculate_health_score(bot_id, bot_status, bot_metrics)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)

    @pytest.mark.asyncio
    async def test_monitoring_loop(self, monitor):
        """Test monitoring loop functionality."""
        await monitor.start()
        
        # Add a bot with metrics
        bot_id = "test_bot_001"
        metrics = BotMetrics(
            cpu_usage=45.0,
            memory_usage=60.0,
            active_positions=3,
            orders_per_minute=5.0,
            error_rate=0.02,
            latency_ms=125.0,
            uptime_seconds=3600,
            total_trades=25,
            successful_trades=23,
            pnl=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc)
        )
        await monitor.update_bot_metrics(bot_id, metrics)
        
        # Run one cycle of monitoring loop
        await monitor._monitoring_loop()
        
        # Should complete without errors
        assert monitor.is_running

    @pytest.mark.asyncio
    async def test_cleanup_old_metrics(self, monitor, bot_metrics):
        """Test cleanup of old metrics."""
        bot_id = "test_bot_001"
        
        # Add old metrics
        old_metrics = BotMetrics(
            cpu_usage=45.0,
            memory_usage=60.0,
            active_positions=3,
            orders_per_minute=5.0,
            error_rate=0.02,
            latency_ms=125.0,
            uptime_seconds=3600,
            total_trades=25,
            successful_trades=23,
            pnl=Decimal("150.50"),
            timestamp=datetime.now(timezone.utc) - timedelta(days=2)
        )
        
        # Simulate old metrics in history
        monitor.metrics_history[bot_id] = [old_metrics] * 100
        
        # Run cleanup
        await monitor._cleanup_old_metrics()
        
        # Should have cleaned up old metrics
        assert len(monitor.metrics_history.get(bot_id, [])) < 100

    @pytest.mark.asyncio
    async def test_export_metrics_influxdb_disabled(self, monitor, bot_metrics):
        """Test metrics export when InfluxDB is disabled."""
        bot_id = "test_bot_001"
        
        # Should handle gracefully when InfluxDB is disabled
        await monitor._export_metrics_to_influxdb(bot_id, bot_metrics)
        
        # No exception should be raised

    @pytest.mark.asyncio
    async def test_export_metrics_influxdb_enabled(self, monitor, bot_metrics):
        """Test metrics export when InfluxDB is enabled."""
        monitor.config.monitoring.influxdb_enabled = True
        bot_id = "test_bot_001"
        
        with patch('src.bot_management.bot_monitor.InfluxDBClient') as mock_client:
            mock_write_api = AsyncMock()
            mock_client.return_value.write_api.return_value = mock_write_api
            
            await monitor._export_metrics_to_influxdb(bot_id, bot_metrics)
            
            # Should have attempted to write to InfluxDB
            mock_write_api.write.assert_called()

    @pytest.mark.asyncio
    async def test_alert_rate_limiting(self, monitor):
        """Test alert rate limiting functionality."""
        bot_id = "test_bot_001"
        alert_type = "high_cpu_usage"
        
        # Generate multiple alerts of same type
        for i in range(5):
            await monitor._generate_alert(bot_id, alert_type, "warning", "Test alert")
        
        # Should have rate limited duplicate alerts
        same_type_alerts = [
            alert for alert in monitor.alert_history 
            if alert["alert_type"] == alert_type and alert["bot_id"] == bot_id
        ]
        assert len(same_type_alerts) < 5  # Should be rate limited

    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self, monitor):
        """Test performance degradation detection."""
        bot_id = "test_bot_001"
        
        # Establish good baseline
        for i in range(10):
            good_metrics = BotMetrics(
                cpu_usage=30.0,
                memory_usage=40.0,
                active_positions=3,
                orders_per_minute=10.0,
                error_rate=0.01,
                latency_ms=100.0,
                uptime_seconds=3600,
                total_trades=50,
                successful_trades=49,
                pnl=Decimal("200.00"),
                timestamp=datetime.now(timezone.utc)
            )
            await monitor.update_bot_metrics(bot_id, good_metrics)
        
        await monitor._establish_performance_baseline(bot_id)
        
        # Add degraded performance metrics
        degraded_metrics = BotMetrics(
            cpu_usage=70.0,  # Higher CPU
            memory_usage=80.0,  # Higher memory
            active_positions=3,
            orders_per_minute=2.0,  # Much lower throughput
            error_rate=0.08,  # Higher error rate
            latency_ms=300.0,  # Higher latency
            uptime_seconds=3600,
            total_trades=55,
            successful_trades=50,
            pnl=Decimal("150.00"),  # Lower performance
            timestamp=datetime.now(timezone.utc)
        )
        
        degradation = await monitor._detect_performance_degradation(bot_id, degraded_metrics)
        
        assert degradation["is_degraded"]
        assert len(degradation["degraded_metrics"]) > 0

    @pytest.mark.asyncio
    async def test_resource_usage_monitoring(self, monitor, bot_metrics):
        """Test resource usage monitoring."""
        bot_id = "test_bot_001"
        
        await monitor.update_bot_metrics(bot_id, bot_metrics)
        
        resource_usage = await monitor.get_resource_usage_summary(bot_id)
        
        assert "cpu_usage" in resource_usage
        assert "memory_usage" in resource_usage
        assert "current" in resource_usage["cpu_usage"]
        assert "trend" in resource_usage["cpu_usage"]

    @pytest.mark.asyncio
    async def test_predictive_alerts(self, monitor):
        """Test predictive alert generation."""
        bot_id = "test_bot_001"
        
        # Create trending metrics that suggest future problems
        base_time = datetime.now(timezone.utc)
        for i in range(10):
            # Gradually increasing CPU usage
            trending_metrics = BotMetrics(
                cpu_usage=50.0 + i * 3,  # Trending upward
                memory_usage=60.0 + i * 2,
                active_positions=3,
                orders_per_minute=5.0 - i * 0.2,  # Trending downward
                error_rate=0.02 + i * 0.01,  # Trending upward
                latency_ms=125.0 + i * 10,
                uptime_seconds=3600,
                total_trades=25,
                successful_trades=23,
                pnl=Decimal("150.50"),
                timestamp=base_time + timedelta(minutes=i * 5)
            )
            await monitor.update_bot_metrics(bot_id, trending_metrics)
        
        # Check for predictive alerts
        predictions = await monitor._generate_predictive_alerts(bot_id)
        
        assert isinstance(predictions, list)
        # Should detect upward CPU trend

    @pytest.mark.asyncio
    async def test_bot_comparison_analysis(self, monitor):
        """Test bot comparison analysis."""
        # Add metrics for multiple bots
        bots_data = [
            ("high_performer", 25.0, 40.0, 10.0, 0.01),
            ("average_performer", 50.0, 60.0, 5.0, 0.03),
            ("low_performer", 80.0, 85.0, 2.0, 0.08)
        ]
        
        for bot_id, cpu, memory, orders, error_rate in bots_data:
            metrics = BotMetrics(
                cpu_usage=cpu,
                memory_usage=memory,
                active_positions=3,
                orders_per_minute=orders,
                error_rate=error_rate,
                latency_ms=125.0,
                uptime_seconds=3600,
                total_trades=25,
                successful_trades=23,
                pnl=Decimal("150.50"),
                timestamp=datetime.now(timezone.utc)
            )
            await monitor.update_bot_metrics(bot_id, metrics)
        
        comparison = await monitor.compare_bot_performance()
        
        assert "rankings" in comparison
        assert "performance_gaps" in comparison
        assert len(comparison["rankings"]) == 3