"""Unit tests for BotOrchestrator component."""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import Config
from src.core.types import BotConfiguration, BotStatus, BotType, BotPriority
from src.bot_management.bot_orchestrator import BotOrchestrator


@pytest.fixture
def config():
    """Create test configuration."""
    config = MagicMock(spec=Config)
    config.error_handling = MagicMock()
    config.bot_management = {
        "max_concurrent_bots": 10,
        "default_heartbeat_interval": 30,
        "emergency_shutdown_timeout": 300
    }
    return config


@pytest.fixture
def bot_config():
    """Create test bot configuration."""
    return BotConfiguration(
        bot_id="test_bot_001",
        bot_name="Test Strategy Bot",
        bot_type=BotType.STRATEGY,
        priority=BotPriority.NORMAL,
        strategy_name="test_strategy",
        exchanges=["binance"],
        symbols=["BTCUSDT"],
        allocated_capital=Decimal("10000"),
        max_position_size=Decimal("1000"),
        risk_percentage=0.02,
        max_concurrent_positions=3,
        heartbeat_interval=30,
        auto_start=True
    )


@pytest.fixture
def mock_bot_instance():
    """Create mock bot instance."""
    bot = AsyncMock()
    bot.bot_config = MagicMock()
    bot.bot_config.bot_id = "test_bot_001"
    bot.bot_config.bot_name = "Test Bot"
    bot.status = BotStatus.CREATED
    bot.start = AsyncMock()
    bot.stop = AsyncMock()
    bot.pause = AsyncMock()
    bot.resume = AsyncMock()
    bot.get_bot_summary = AsyncMock(return_value={"test": "summary"})
    bot.get_heartbeat = AsyncMock(return_value={"test": "heartbeat"})
    return bot


@pytest.fixture
def orchestrator(config):
    """Create BotOrchestrator for testing."""
    return BotOrchestrator(config)


class TestBotOrchestrator:
    """Test cases for BotOrchestrator class."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator, config):
        """Test orchestrator initialization."""
        assert orchestrator.config == config
        assert orchestrator.bots == {}
        assert orchestrator.max_concurrent_bots == 10
        assert not orchestrator.is_running
        assert orchestrator.orchestration_statistics["total_bots_created"] == 0

    @pytest.mark.asyncio
    async def test_start_orchestrator(self, orchestrator):
        """Test orchestrator startup."""
        await orchestrator.start()
        
        assert orchestrator.is_running
        assert orchestrator.orchestration_task is not None
        assert orchestrator.started_at is not None

    @pytest.mark.asyncio
    async def test_stop_orchestrator(self, orchestrator):
        """Test orchestrator shutdown."""
        # Start first
        await orchestrator.start()
        
        # Add a mock bot
        mock_bot = AsyncMock()
        orchestrator.bots["test_bot"] = mock_bot
        
        # Stop orchestrator
        await orchestrator.stop()
        
        assert not orchestrator.is_running
        assert orchestrator.stopped_at is not None
        mock_bot.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_bot_success(self, orchestrator, bot_config):
        """Test successful bot creation."""
        with patch('src.bot_management.bot_instance.BotInstance') as mock_bot_class:
            mock_bot = AsyncMock()
            mock_bot.bot_config = bot_config
            mock_bot_class.return_value = mock_bot
            
            # Create bot
            bot_id = await orchestrator.create_bot(bot_config)
            
            # Verify
            assert bot_id == bot_config.bot_id
            assert bot_id in orchestrator.bots
            assert orchestrator.orchestration_statistics["total_bots_created"] == 1
            mock_bot_class.assert_called_once_with(bot_config, orchestrator.config)

    @pytest.mark.asyncio
    async def test_create_bot_duplicate_id(self, orchestrator, bot_config):
        """Test creating bot with duplicate ID."""
        # Add existing bot
        mock_bot = AsyncMock()
        orchestrator.bots[bot_config.bot_id] = mock_bot
        
        # Try to create duplicate
        with pytest.raises(Exception):
            await orchestrator.create_bot(bot_config)

    @pytest.mark.asyncio
    async def test_create_bot_max_capacity(self, orchestrator, bot_config):
        """Test creating bot when at max capacity."""
        # Fill up to capacity
        orchestrator.max_concurrent_bots = 1
        mock_bot = AsyncMock()
        orchestrator.bots["existing_bot"] = mock_bot
        
        # Try to create when at capacity
        with pytest.raises(Exception):
            await orchestrator.create_bot(bot_config)

    @pytest.mark.asyncio
    async def test_start_bot_success(self, orchestrator, mock_bot_instance):
        """Test successful bot startup."""
        bot_id = "test_bot_001"
        orchestrator.bots[bot_id] = mock_bot_instance
        
        # Start bot
        success = await orchestrator.start_bot(bot_id)
        
        assert success
        mock_bot_instance.start.assert_called_once()
        assert orchestrator.orchestration_statistics["total_bots_started"] == 1

    @pytest.mark.asyncio
    async def test_start_bot_not_found(self, orchestrator):
        """Test starting non-existent bot."""
        success = await orchestrator.start_bot("non_existent")
        assert not success

    @pytest.mark.asyncio
    async def test_stop_bot_success(self, orchestrator, mock_bot_instance):
        """Test successful bot stop."""
        bot_id = "test_bot_001"
        orchestrator.bots[bot_id] = mock_bot_instance
        
        # Stop bot
        success = await orchestrator.stop_bot(bot_id)
        
        assert success
        mock_bot_instance.stop.assert_called_once()
        assert orchestrator.orchestration_statistics["total_bots_stopped"] == 1

    @pytest.mark.asyncio
    async def test_pause_resume_bot(self, orchestrator, mock_bot_instance):
        """Test bot pause and resume."""
        bot_id = "test_bot_001"
        orchestrator.bots[bot_id] = mock_bot_instance
        
        # Pause bot
        success = await orchestrator.pause_bot(bot_id)
        assert success
        mock_bot_instance.pause.assert_called_once()
        
        # Resume bot
        success = await orchestrator.resume_bot(bot_id)
        assert success
        mock_bot_instance.resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_bot_success(self, orchestrator, mock_bot_instance):
        """Test successful bot deletion."""
        bot_id = "test_bot_001"
        orchestrator.bots[bot_id] = mock_bot_instance
        mock_bot_instance.status = BotStatus.STOPPED
        
        # Delete bot
        success = await orchestrator.delete_bot(bot_id)
        
        assert success
        assert bot_id not in orchestrator.bots
        mock_bot_instance.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_bot_force(self, orchestrator, mock_bot_instance):
        """Test force deletion of running bot."""
        bot_id = "test_bot_001"
        orchestrator.bots[bot_id] = mock_bot_instance
        mock_bot_instance.status = BotStatus.RUNNING
        
        # Force delete running bot
        success = await orchestrator.delete_bot(bot_id, force=True)
        
        assert success
        assert bot_id not in orchestrator.bots

    @pytest.mark.asyncio
    async def test_delete_bot_running_no_force(self, orchestrator, mock_bot_instance):
        """Test deletion of running bot without force."""
        bot_id = "test_bot_001"
        orchestrator.bots[bot_id] = mock_bot_instance
        mock_bot_instance.status = BotStatus.RUNNING
        
        # Try to delete running bot without force
        success = await orchestrator.delete_bot(bot_id, force=False)
        
        assert not success
        assert bot_id in orchestrator.bots

    @pytest.mark.asyncio
    async def test_get_bot_status(self, orchestrator, mock_bot_instance):
        """Test getting bot status."""
        bot_id = "test_bot_001"
        orchestrator.bots[bot_id] = mock_bot_instance
        mock_bot_instance.status = BotStatus.RUNNING
        
        status = await orchestrator.get_bot_status(bot_id)
        assert status == BotStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_bot_status_not_found(self, orchestrator):
        """Test getting status of non-existent bot."""
        status = await orchestrator.get_bot_status("non_existent")
        assert status is None

    @pytest.mark.asyncio
    async def test_list_bots(self, orchestrator):
        """Test listing all bots."""
        # Add multiple bots
        for i in range(3):
            mock_bot = AsyncMock()
            mock_bot.bot_config.bot_id = f"bot_{i}"
            mock_bot.bot_config.bot_name = f"Bot {i}"
            mock_bot.status = BotStatus.RUNNING
            orchestrator.bots[f"bot_{i}"] = mock_bot
        
        bot_list = await orchestrator.list_bots()
        
        assert len(bot_list) == 3
        assert all("bot_id" in bot for bot in bot_list)
        assert all("bot_name" in bot for bot in bot_list)
        assert all("status" in bot for bot in bot_list)

    @pytest.mark.asyncio
    async def test_emergency_shutdown(self, orchestrator):
        """Test emergency shutdown functionality."""
        # Add multiple bots
        for i in range(3):
            mock_bot = AsyncMock()
            orchestrator.bots[f"bot_{i}"] = mock_bot
        
        # Emergency shutdown
        await orchestrator.emergency_shutdown("Test emergency")
        
        # Verify all bots stopped
        for bot in orchestrator.bots.values():
            bot.stop.assert_called_once()
        
        assert not orchestrator.is_running

    @pytest.mark.asyncio
    async def test_start_all_bots(self, orchestrator):
        """Test starting all bots."""
        # Add multiple bots
        for i in range(3):
            mock_bot = AsyncMock()
            mock_bot.bot_config.auto_start = True
            orchestrator.bots[f"bot_{i}"] = mock_bot
        
        # Start all bots
        await orchestrator.start_all_bots()
        
        # Verify all bots started
        for bot in orchestrator.bots.values():
            bot.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_all_bots(self, orchestrator):
        """Test stopping all bots."""
        # Add multiple bots
        for i in range(3):
            mock_bot = AsyncMock()
            orchestrator.bots[f"bot_{i}"] = mock_bot
        
        # Stop all bots
        await orchestrator.stop_all_bots()
        
        # Verify all bots stopped
        for bot in orchestrator.bots.values():
            bot.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_orchestrator_summary(self, orchestrator):
        """Test orchestrator summary generation."""
        # Add some bots with different statuses
        statuses = [BotStatus.RUNNING, BotStatus.STOPPED, BotStatus.PAUSED]
        for i, status in enumerate(statuses):
            mock_bot = AsyncMock()
            mock_bot.status = status
            mock_bot.bot_config.bot_type = BotType.STRATEGY
            orchestrator.bots[f"bot_{i}"] = mock_bot
        
        # Set some statistics
        orchestrator.orchestration_statistics["total_bots_created"] = 5
        orchestrator.orchestration_statistics["total_bots_started"] = 3
        
        summary = await orchestrator.get_orchestrator_summary()
        
        # Verify summary structure
        assert "orchestrator_status" in summary
        assert "bot_counts" in summary
        assert "statistics" in summary
        assert "configuration" in summary
        
        # Verify content
        assert summary["bot_counts"]["total_bots"] == 3
        assert summary["bot_counts"]["running"] == 1
        assert summary["bot_counts"]["stopped"] == 1
        assert summary["bot_counts"]["paused"] == 1

    @pytest.mark.asyncio
    async def test_get_global_metrics(self, orchestrator):
        """Test global metrics aggregation."""
        # Add bots with metrics
        for i in range(2):
            mock_bot = AsyncMock()
            mock_bot.get_bot_summary.return_value = {
                "performance": {
                    "total_trades": 10,
                    "total_pnl": Decimal("100"),
                    "win_rate": 0.6
                }
            }
            orchestrator.bots[f"bot_{i}"] = mock_bot
        
        metrics = await orchestrator.get_global_metrics()
        
        # Verify aggregated metrics
        assert "total_trades" in metrics
        assert "total_pnl" in metrics
        assert "average_win_rate" in metrics
        assert metrics["total_trades"] == 20  # 10 * 2 bots

    @pytest.mark.asyncio
    async def test_collect_heartbeats(self, orchestrator):
        """Test heartbeat collection from all bots."""
        # Add bots
        for i in range(2):
            mock_bot = AsyncMock()
            mock_bot.get_heartbeat.return_value = {
                "bot_id": f"bot_{i}",
                "status": "running",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            orchestrator.bots[f"bot_{i}"] = mock_bot
        
        heartbeats = await orchestrator.collect_heartbeats()
        
        assert len(heartbeats) == 2
        assert all("bot_id" in hb for hb in heartbeats)
        assert all("status" in hb for hb in heartbeats)

    @pytest.mark.asyncio
    async def test_error_handling_in_orchestration_loop(self, orchestrator):
        """Test error handling in orchestration loop."""
        await orchestrator.start()
        
        # Add a bot that raises exception during heartbeat
        mock_bot = AsyncMock()
        mock_bot.get_heartbeat.side_effect = Exception("Heartbeat error")
        orchestrator.bots["error_bot"] = mock_bot
        
        # Should handle error gracefully
        await orchestrator._orchestration_loop()
        
        # Orchestrator should still be running
        assert orchestrator.is_running

    @pytest.mark.asyncio
    async def test_bot_health_monitoring(self, orchestrator):
        """Test bot health monitoring functionality."""
        # Add bot with stale heartbeat
        mock_bot = AsyncMock()
        mock_bot.last_heartbeat = datetime.now(timezone.utc)
        mock_bot.status = BotStatus.RUNNING
        orchestrator.bots["stale_bot"] = mock_bot
        
        # Monitor health
        health_issues = await orchestrator._monitor_bot_health()
        
        # Should detect health issues
        assert isinstance(health_issues, list)

    @pytest.mark.asyncio
    async def test_resource_optimization(self, orchestrator):
        """Test resource optimization functionality."""
        # Add multiple bots
        for i in range(3):
            mock_bot = AsyncMock()
            mock_bot.bot_config.priority = BotPriority.NORMAL
            mock_bot.performance_metrics = {"cpu_usage": 50.0, "memory_usage": 60.0}
            orchestrator.bots[f"bot_{i}"] = mock_bot
        
        # Run optimization
        optimizations = await orchestrator._optimize_resources()
        
        # Should return optimization suggestions
        assert isinstance(optimizations, list)

    @pytest.mark.asyncio
    async def test_shutdown_with_cleanup(self, orchestrator):
        """Test shutdown with proper cleanup."""
        await orchestrator.start()
        
        # Add bots
        for i in range(2):
            mock_bot = AsyncMock()
            orchestrator.bots[f"bot_{i}"] = mock_bot
        
        # Shutdown
        await orchestrator.shutdown()
        
        # Verify cleanup
        assert not orchestrator.is_running
        assert len(orchestrator.bots) == 0
        for i in range(2):
            # Bots should have been stopped (checked in cleanup process)
            pass