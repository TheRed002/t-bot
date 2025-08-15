"""Unit tests for BotLifecycle component."""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import Config
from src.core.types import BotConfiguration, BotType, BotPriority
from src.bot_management.bot_lifecycle import BotLifecycle


@pytest.fixture
def config():
    """Create test configuration."""
    config = MagicMock(spec=Config)
    config.error_handling = MagicMock()
    config.bot_management = {
        "event_retention_hours": 168,
        "graceful_shutdown_timeout": 300,
        "restart_max_attempts": 3,
        "restart_delay_seconds": 60
    }
    return config


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator."""
    orchestrator = AsyncMock()
    orchestrator.create_bot = AsyncMock(return_value="test_bot_001")
    orchestrator.start_bot = AsyncMock(return_value=True)
    orchestrator.stop_bot = AsyncMock(return_value=True)
    orchestrator.pause_bot = AsyncMock(return_value=True)
    orchestrator.delete_bot = AsyncMock(return_value=True)
    return orchestrator


@pytest.fixture
def lifecycle(config):
    """Create BotLifecycle for testing."""
    return BotLifecycle(config)


class TestBotLifecycle:
    """Test cases for BotLifecycle class."""

    @pytest.mark.asyncio
    async def test_lifecycle_initialization(self, lifecycle, config):
        """Test lifecycle manager initialization."""
        assert lifecycle.config == config
        assert lifecycle.bot_lifecycles == {}
        assert lifecycle.bot_templates != {}  # Should have built-in templates
        assert lifecycle.deployment_strategies != {}
        assert not lifecycle.is_running
        assert lifecycle.lifecycle_stats["bots_created"] == 0

    @pytest.mark.asyncio
    async def test_start_lifecycle_manager(self, lifecycle):
        """Test lifecycle manager startup."""
        await lifecycle.start()
        
        assert lifecycle.is_running
        assert lifecycle.lifecycle_task is not None

    @pytest.mark.asyncio
    async def test_stop_lifecycle_manager(self, lifecycle):
        """Test lifecycle manager shutdown."""
        await lifecycle.start()
        await lifecycle.stop()
        
        assert not lifecycle.is_running

    @pytest.mark.asyncio
    async def test_create_bot_from_template_success(self, lifecycle):
        """Test successful bot creation from template."""
        template_name = "simple_strategy_bot"
        bot_name = "Test Strategy Bot"
        custom_config = {
            "strategy_name": "test_strategy",
            "exchanges": ["binance"],
            "symbols": ["BTCUSDT"],
            "allocated_capital": Decimal("10000")
        }
        
        bot_config = await lifecycle.create_bot_from_template(
            template_name, bot_name, custom_config
        )
        
        assert bot_config is not None
        assert bot_config.bot_name == bot_name
        assert bot_config.strategy_name == "test_strategy"
        assert bot_config.bot_type == BotType.STRATEGY
        assert lifecycle.lifecycle_stats["bots_created"] == 1
        assert bot_config.bot_id in lifecycle.bot_lifecycles

    @pytest.mark.asyncio
    async def test_create_bot_invalid_template(self, lifecycle):
        """Test bot creation with invalid template."""
        with pytest.raises(Exception):
            await lifecycle.create_bot_from_template(
                "non_existent_template", "Test Bot", {}
            )

    @pytest.mark.asyncio
    async def test_create_bot_missing_required_fields(self, lifecycle):
        """Test bot creation with missing required fields."""
        template_name = "simple_strategy_bot"
        custom_config = {
            "strategy_name": "test_strategy"
            # Missing required fields: exchanges, symbols, allocated_capital
        }
        
        with pytest.raises(Exception):
            await lifecycle.create_bot_from_template(
                template_name, "Test Bot", custom_config
            )

    @pytest.mark.asyncio
    async def test_deploy_bot_immediate_strategy(self, lifecycle, mock_orchestrator):
        """Test bot deployment with immediate strategy."""
        # Create bot configuration
        bot_config = BotConfiguration(
            bot_id="test_bot_001",
            bot_name="Test Bot",
            bot_type=BotType.STRATEGY,
            priority=BotPriority.NORMAL,
            strategy_name="test_strategy",
            exchanges=["binance"],
            symbols=["BTCUSDT"],
            allocated_capital=Decimal("10000"),
            max_position_size=Decimal("1000"),
            auto_start=True
        )
        
        # Initialize lifecycle tracking
        await lifecycle._initialize_bot_lifecycle("test_bot_001", "simple_strategy_bot", "immediate")
        
        # Deploy bot
        success = await lifecycle.deploy_bot(bot_config, mock_orchestrator)
        
        assert success
        mock_orchestrator.create_bot.assert_called_once()
        mock_orchestrator.start_bot.assert_called_once()
        assert lifecycle.lifecycle_stats["bots_deployed"] == 1

    @pytest.mark.asyncio
    async def test_deploy_bot_staged_strategy(self, lifecycle, mock_orchestrator):
        """Test bot deployment with staged strategy."""
        bot_config = BotConfiguration(
            bot_id="test_bot_001",
            bot_name="Test Bot",
            bot_type=BotType.STRATEGY,
            priority=BotPriority.NORMAL,
            strategy_name="test_strategy",
            exchanges=["binance"],
            symbols=["BTCUSDT"],
            allocated_capital=Decimal("10000"),
            max_position_size=Decimal("1000"),
            auto_start=True
        )
        
        await lifecycle._initialize_bot_lifecycle("test_bot_001", "simple_strategy_bot", "staged")
        
        # Mock fast deployment for testing
        with patch('asyncio.sleep', new_callable=AsyncMock):
            success = await lifecycle.deploy_bot(bot_config, mock_orchestrator)
        
        assert success
        mock_orchestrator.create_bot.assert_called_once()

    @pytest.mark.asyncio
    async def test_deploy_bot_failure(self, lifecycle, mock_orchestrator):
        """Test bot deployment failure."""
        bot_config = BotConfiguration(
            bot_id="test_bot_001",
            bot_name="Test Bot",
            bot_type=BotType.STRATEGY,
            priority=BotPriority.NORMAL,
            strategy_name="test_strategy",
            exchanges=["binance"],
            symbols=["BTCUSDT"],
            allocated_capital=Decimal("10000"),
            max_position_size=Decimal("1000"),
            auto_start=True
        )
        
        # Mock orchestrator failure
        mock_orchestrator.create_bot.side_effect = Exception("Deployment failed")
        
        await lifecycle._initialize_bot_lifecycle("test_bot_001", "simple_strategy_bot", "immediate")
        
        with pytest.raises(Exception):
            await lifecycle.deploy_bot(bot_config, mock_orchestrator)
        
        assert lifecycle.lifecycle_stats["failed_deployments"] == 1

    @pytest.mark.asyncio
    async def test_terminate_bot_graceful(self, lifecycle, mock_orchestrator):
        """Test graceful bot termination."""
        bot_id = "test_bot_001"
        
        # Initialize lifecycle tracking
        await lifecycle._initialize_bot_lifecycle(bot_id, "simple_strategy_bot", "immediate")
        
        # Mock fast termination for testing
        with patch('asyncio.sleep', new_callable=AsyncMock):
            success = await lifecycle.terminate_bot(bot_id, mock_orchestrator, graceful=True)
        
        assert success
        mock_orchestrator.pause_bot.assert_called_once()
        mock_orchestrator.stop_bot.assert_called_once()
        mock_orchestrator.delete_bot.assert_called_once()
        assert lifecycle.lifecycle_stats["bots_terminated"] == 1

    @pytest.mark.asyncio
    async def test_terminate_bot_immediate(self, lifecycle, mock_orchestrator):
        """Test immediate bot termination."""
        bot_id = "test_bot_001"
        
        await lifecycle._initialize_bot_lifecycle(bot_id, "simple_strategy_bot", "immediate")
        
        success = await lifecycle.terminate_bot(bot_id, mock_orchestrator, graceful=False)
        
        assert success
        mock_orchestrator.stop_bot.assert_called_once()
        # Pause should not be called for immediate termination

    @pytest.mark.asyncio
    async def test_restart_bot_success(self, lifecycle, mock_orchestrator):
        """Test successful bot restart."""
        bot_id = "test_bot_001"
        
        # Mock fast restart for testing
        with patch('asyncio.sleep', new_callable=AsyncMock):
            success = await lifecycle.restart_bot(bot_id, mock_orchestrator)
        
        assert success
        mock_orchestrator.stop_bot.assert_called_once()
        mock_orchestrator.start_bot.assert_called_once()
        assert lifecycle.lifecycle_stats["successful_restarts"] == 1

    @pytest.mark.asyncio
    async def test_restart_bot_failure(self, lifecycle, mock_orchestrator):
        """Test bot restart failure."""
        bot_id = "test_bot_001"
        
        # Mock restart failure
        mock_orchestrator.start_bot.return_value = False
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            success = await lifecycle.restart_bot(bot_id, mock_orchestrator)
        
        assert not success
        assert lifecycle.lifecycle_stats["failed_restarts"] == 1

    @pytest.mark.asyncio
    async def test_get_lifecycle_summary(self, lifecycle):
        """Test lifecycle summary generation."""
        # Add some lifecycle data
        await lifecycle._initialize_bot_lifecycle("bot1", "simple_strategy_bot", "immediate")
        await lifecycle._initialize_bot_lifecycle("bot2", "arbitrage_bot", "staged")
        
        lifecycle.lifecycle_stats["bots_created"] = 5
        lifecycle.lifecycle_stats["bots_deployed"] = 4
        lifecycle.lifecycle_stats["bots_terminated"] = 1
        
        summary = await lifecycle.get_lifecycle_summary()
        
        # Verify summary structure
        assert "lifecycle_overview" in summary
        assert "deployment_states" in summary
        assert "termination_states" in summary
        assert "lifecycle_statistics" in summary
        assert "recent_events_24h" in summary
        assert "templates" in summary
        
        # Verify content
        assert summary["lifecycle_overview"]["managed_bots"] == 2
        assert summary["lifecycle_overview"]["available_templates"] == len(lifecycle.bot_templates)
        assert summary["lifecycle_statistics"]["bots_created"] == 5

    @pytest.mark.asyncio
    async def test_get_bot_lifecycle_details(self, lifecycle):
        """Test bot lifecycle details retrieval."""
        bot_id = "test_bot_001"
        
        # Initialize lifecycle and add some events
        await lifecycle._initialize_bot_lifecycle(bot_id, "simple_strategy_bot", "immediate")
        await lifecycle._record_lifecycle_event(bot_id, "created", {"template": "simple_strategy_bot"})
        await lifecycle._record_lifecycle_event(bot_id, "deployed", {"strategy": "immediate"})
        
        details = await lifecycle.get_bot_lifecycle_details(bot_id)
        
        assert details is not None
        assert details["bot_id"] == bot_id
        assert "lifecycle_data" in details
        assert "events" in details
        assert details["event_count"] == 2

    @pytest.mark.asyncio
    async def test_get_bot_lifecycle_details_not_found(self, lifecycle):
        """Test lifecycle details for non-existent bot."""
        details = await lifecycle.get_bot_lifecycle_details("non_existent")
        assert details is None

    @pytest.mark.asyncio
    async def test_built_in_templates(self, lifecycle):
        """Test built-in bot templates."""
        expected_templates = [
            "simple_strategy_bot",
            "arbitrage_bot", 
            "market_maker_bot",
            "hybrid_strategy_bot",
            "scanner_bot"
        ]
        
        for template_name in expected_templates:
            assert template_name in lifecycle.bot_templates
            template = lifecycle.bot_templates[template_name]
            assert "name" in template
            assert "description" in template
            assert "default_config" in template
            assert "required_fields" in template

    @pytest.mark.asyncio
    async def test_deployment_strategies(self, lifecycle):
        """Test deployment strategies availability."""
        expected_strategies = [
            "immediate",
            "staged",
            "blue_green",
            "canary", 
            "rolling"
        ]
        
        for strategy in expected_strategies:
            assert strategy in lifecycle.deployment_strategies

    @pytest.mark.asyncio
    async def test_lifecycle_event_recording(self, lifecycle):
        """Test lifecycle event recording."""
        bot_id = "test_bot_001"
        event_type = "test_event"
        event_data = {"test": "data"}
        
        await lifecycle._record_lifecycle_event(bot_id, event_type, event_data)
        
        # Should have recorded event
        assert len(lifecycle.lifecycle_events) == 1
        event = lifecycle.lifecycle_events[0]
        assert event["bot_id"] == bot_id
        assert event["event_type"] == event_type
        assert event["event_data"] == event_data

    @pytest.mark.asyncio
    async def test_lifecycle_event_cleanup(self, lifecycle):
        """Test lifecycle event cleanup."""
        # Add many old events
        for i in range(10):
            event = {
                "event_id": f"event_{i}",
                "bot_id": f"bot_{i}",
                "event_type": "test",
                "event_data": {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            lifecycle.lifecycle_events.append(event)
        
        # Record new event (which triggers cleanup)
        await lifecycle._record_lifecycle_event("new_bot", "new_event", {})
        
        # Should have all events since they're recent
        assert len(lifecycle.lifecycle_events) == 11

    @pytest.mark.asyncio
    async def test_lifecycle_monitoring_loop(self, lifecycle):
        """Test lifecycle monitoring loop."""
        await lifecycle.start()
        
        # Add some lifecycle data
        await lifecycle._initialize_bot_lifecycle("test_bot", "simple_strategy_bot", "immediate")
        
        # Run one cycle of monitoring loop
        await lifecycle._lifecycle_loop()
        
        # Should complete without errors
        assert lifecycle.is_running

    @pytest.mark.asyncio
    async def test_template_validation(self, lifecycle):
        """Test bot template validation."""
        # Test all built-in templates have required structure
        for template_name, template in lifecycle.bot_templates.items():
            # Verify required template fields
            assert "name" in template
            assert "description" in template
            assert "default_config" in template
            assert "required_fields" in template
            assert "optional_fields" in template
            
            # Verify default config has bot_type
            assert "bot_type" in template["default_config"]
            assert isinstance(template["default_config"]["bot_type"], BotType)
            
            # Verify required fields is a list
            assert isinstance(template["required_fields"], list)
            assert isinstance(template["optional_fields"], list)

    @pytest.mark.asyncio
    async def test_deployment_strategy_execution(self, lifecycle, mock_orchestrator):
        """Test different deployment strategy execution."""
        bot_config = BotConfiguration(
            bot_id="test_bot_001",
            bot_name="Test Bot",
            bot_type=BotType.STRATEGY,
            priority=BotPriority.NORMAL,
            strategy_name="test_strategy",
            exchanges=["binance"],
            symbols=["BTCUSDT"],
            allocated_capital=Decimal("10000"),
            max_position_size=Decimal("1000"),
            auto_start=True
        )
        
        strategies_to_test = ["immediate", "staged", "blue_green", "canary", "rolling"]
        
        for strategy in strategies_to_test:
            # Test each deployment strategy
            deployment_func = lifecycle.deployment_strategies[strategy]
            
            with patch('asyncio.sleep', new_callable=AsyncMock):
                success = await deployment_func(bot_config, mock_orchestrator, {})
            
            assert success is not None
            assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_graceful_vs_immediate_termination(self, lifecycle, mock_orchestrator):
        """Test difference between graceful and immediate termination."""
        bot_id = "test_bot_001"
        
        await lifecycle._initialize_bot_lifecycle(bot_id, "simple_strategy_bot", "immediate")
        
        # Test graceful termination
        with patch('asyncio.sleep', new_callable=AsyncMock):
            graceful_success = await lifecycle._graceful_termination(bot_id, mock_orchestrator)
        
        # Reset mock
        mock_orchestrator.reset_mock()
        
        # Test immediate termination
        immediate_success = await lifecycle._immediate_termination(bot_id, mock_orchestrator)
        
        # Both should succeed but with different call patterns
        assert graceful_success
        assert immediate_success

    @pytest.mark.asyncio
    async def test_error_handling_in_deployment(self, lifecycle, mock_orchestrator):
        """Test error handling during deployment."""
        bot_config = BotConfiguration(
            bot_id="test_bot_001",
            bot_name="Test Bot",
            bot_type=BotType.STRATEGY,
            priority=BotPriority.NORMAL,
            strategy_name="test_strategy",
            exchanges=["binance"],
            symbols=["BTCUSDT"],
            allocated_capital=Decimal("10000"),
            max_position_size=Decimal("1000"),
            auto_start=True
        )
        
        # Mock orchestrator to fail
        mock_orchestrator.create_bot.side_effect = Exception("Create failed")
        
        # Should handle deployment failure gracefully
        success = await lifecycle._deploy_immediate(bot_config, mock_orchestrator, {})
        assert not success

    @pytest.mark.asyncio
    async def test_lifecycle_statistics_tracking(self, lifecycle, mock_orchestrator):
        """Test lifecycle statistics tracking."""
        initial_stats = lifecycle.lifecycle_stats.copy()
        
        # Create and deploy a bot
        template_name = "simple_strategy_bot"
        custom_config = {
            "strategy_name": "test_strategy",
            "exchanges": ["binance"],
            "symbols": ["BTCUSDT"],
            "allocated_capital": Decimal("10000")
        }
        
        bot_config = await lifecycle.create_bot_from_template(
            template_name, "Test Bot", custom_config
        )
        
        await lifecycle.deploy_bot(bot_config, mock_orchestrator)
        
        # Verify statistics updated
        assert lifecycle.lifecycle_stats["bots_created"] > initial_stats["bots_created"]
        assert lifecycle.lifecycle_stats["bots_deployed"] > initial_stats["bots_deployed"]
        assert lifecycle.lifecycle_stats["last_lifecycle_action"] is not None

    @pytest.mark.asyncio
    async def test_concurrent_lifecycle_operations(self, lifecycle, mock_orchestrator):
        """Test concurrent lifecycle operations."""
        # Create multiple bots concurrently
        tasks = []
        for i in range(3):
            task = lifecycle.create_bot_from_template(
                "simple_strategy_bot",
                f"Bot {i}",
                {
                    "strategy_name": f"strategy_{i}",
                    "exchanges": ["binance"],
                    "symbols": ["BTCUSDT"],
                    "allocated_capital": Decimal("10000")
                }
            )
            tasks.append(task)
        
        bot_configs = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(bot_configs) == 3
        assert all(config is not None for config in bot_configs)
        assert lifecycle.lifecycle_stats["bots_created"] == 3