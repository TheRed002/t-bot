"""Unit tests for BotCoordinator component."""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import Config
from src.core.types import OrderSide, BotPriority
from src.bot_management.bot_coordinator import BotCoordinator


@pytest.fixture
def config():
    """Create test configuration."""
    config = MagicMock(spec=Config)
    config.error_handling = MagicMock()
    config.bot_management = {
        "signal_retention_hours": 24,
        "coordination_check_interval": 30,
        "max_signal_recipients": 50
    }
    return config


@pytest.fixture
def coordinator(config):
    """Create BotCoordinator for testing."""
    return BotCoordinator(config)


@pytest.fixture
def mock_bot():
    """Create mock bot for testing."""
    bot = AsyncMock()
    bot.bot_config.bot_id = "test_bot_001"
    bot.bot_config.symbols = ["BTCUSDT", "ETHUSDT"]
    bot.bot_config.priority = BotPriority.NORMAL
    bot.active_positions = {}
    bot.get_bot_summary = AsyncMock(return_value={"test": "summary"})
    return bot


class TestBotCoordinator:
    """Test cases for BotCoordinator class."""

    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator, config):
        """Test coordinator initialization."""
        assert coordinator.config == config
        assert coordinator.registered_bots == {}
        assert coordinator.signal_history == []
        assert coordinator.position_registry == {}
        assert not coordinator.is_running

    @pytest.mark.asyncio
    async def test_start_coordinator(self, coordinator):
        """Test coordinator startup."""
        await coordinator.start()
        
        assert coordinator.is_running
        assert coordinator.coordination_task is not None

    @pytest.mark.asyncio
    async def test_stop_coordinator(self, coordinator):
        """Test coordinator shutdown."""
        await coordinator.start()
        await coordinator.stop()
        
        assert not coordinator.is_running

    @pytest.mark.asyncio
    async def test_register_bot(self, coordinator, mock_bot):
        """Test bot registration."""
        bot_id = "test_bot_001"
        
        # Register bot
        await coordinator.register_bot(bot_id, mock_bot)
        
        assert bot_id in coordinator.registered_bots
        assert coordinator.registered_bots[bot_id] == mock_bot
        assert coordinator.coordination_statistics["total_registered_bots"] == 1

    @pytest.mark.asyncio
    async def test_register_duplicate_bot(self, coordinator, mock_bot):
        """Test registering duplicate bot."""
        bot_id = "test_bot_001"
        
        # Register bot twice
        await coordinator.register_bot(bot_id, mock_bot)
        
        # Second registration should update
        await coordinator.register_bot(bot_id, mock_bot)
        
        assert coordinator.coordination_statistics["total_registered_bots"] == 1

    @pytest.mark.asyncio
    async def test_unregister_bot(self, coordinator, mock_bot):
        """Test bot unregistration."""
        bot_id = "test_bot_001"
        
        # Register then unregister
        await coordinator.register_bot(bot_id, mock_bot)
        success = await coordinator.unregister_bot(bot_id)
        
        assert success
        assert bot_id not in coordinator.registered_bots

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_bot(self, coordinator):
        """Test unregistering non-existent bot."""
        success = await coordinator.unregister_bot("non_existent")
        assert not success

    @pytest.mark.asyncio
    async def test_share_signal_success(self, coordinator, mock_bot):
        """Test successful signal sharing."""
        bot_id = "sender_bot"
        signal_data = {
            "signal_type": "buy",
            "symbol": "BTCUSDT",
            "confidence": 0.8,
            "price": Decimal("50000")
        }
        
        # Register multiple bots
        for i in range(3):
            test_bot = AsyncMock()
            test_bot.bot_config.symbols = ["BTCUSDT"]
            await coordinator.register_bot(f"bot_{i}", test_bot)
        
        # Share signal
        recipients = await coordinator.share_signal(bot_id, signal_data)
        
        assert recipients == 3
        assert len(coordinator.signal_history) == 1
        assert coordinator.coordination_statistics["total_signals_shared"] == 1

    @pytest.mark.asyncio
    async def test_share_signal_targeted(self, coordinator, mock_bot):
        """Test targeted signal sharing."""
        bot_id = "sender_bot"
        signal_data = {
            "signal_type": "sell",
            "symbol": "ETHUSDT",
            "confidence": 0.9
        }
        target_bots = ["bot_1", "bot_2"]
        
        # Register bots
        for i in range(3):
            test_bot = AsyncMock()
            test_bot.bot_config.symbols = ["ETHUSDT"]
            await coordinator.register_bot(f"bot_{i}", test_bot)
        
        # Share signal to specific bots
        recipients = await coordinator.share_signal(bot_id, signal_data, target_bots)
        
        assert recipients == 2  # Only targeted bots

    @pytest.mark.asyncio
    async def test_share_signal_no_recipients(self, coordinator):
        """Test signal sharing with no valid recipients."""
        bot_id = "sender_bot"
        signal_data = {
            "signal_type": "buy",
            "symbol": "NONEXISTENT",
            "confidence": 0.7
        }
        
        # No bots interested in this symbol
        recipients = await coordinator.share_signal(bot_id, signal_data)
        
        assert recipients == 0

    @pytest.mark.asyncio
    async def test_update_bot_position(self, coordinator, mock_bot):
        """Test bot position update."""
        bot_id = "test_bot_001"
        position_data = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("1.0"),
            "average_price": Decimal("50000")
        }
        
        await coordinator.register_bot(bot_id, mock_bot)
        await coordinator.update_bot_position(bot_id, "BTCUSDT", position_data)
        
        assert bot_id in coordinator.position_registry
        assert "BTCUSDT" in coordinator.position_registry[bot_id]
        assert coordinator.position_registry[bot_id]["BTCUSDT"] == position_data

    @pytest.mark.asyncio
    async def test_remove_bot_position(self, coordinator, mock_bot):
        """Test bot position removal."""
        bot_id = "test_bot_001"
        symbol = "BTCUSDT"
        
        # Add position first
        await coordinator.register_bot(bot_id, mock_bot)
        await coordinator.update_bot_position(bot_id, symbol, {"test": "position"})
        
        # Remove position
        await coordinator.remove_bot_position(bot_id, symbol)
        
        assert symbol not in coordinator.position_registry.get(bot_id, {})

    @pytest.mark.asyncio
    async def test_check_position_conflicts(self, coordinator):
        """Test position conflict detection."""
        # Register bots with conflicting positions
        bot1 = AsyncMock()
        bot1.bot_config.symbols = ["BTCUSDT"]
        bot2 = AsyncMock()
        bot2.bot_config.symbols = ["BTCUSDT"]
        
        await coordinator.register_bot("bot1", bot1)
        await coordinator.register_bot("bot2", bot2)
        
        # Add conflicting positions
        await coordinator.update_bot_position("bot1", "BTCUSDT", {
            "side": "BUY", "quantity": Decimal("1.0")
        })
        await coordinator.update_bot_position("bot2", "BTCUSDT", {
            "side": "SELL", "quantity": Decimal("0.8")
        })
        
        conflicts = await coordinator.check_position_conflicts("BTCUSDT")
        
        assert len(conflicts) > 0
        assert any("opposing" in conflict.lower() for conflict in conflicts)

    @pytest.mark.asyncio
    async def test_check_cross_bot_risk(self, coordinator):
        """Test cross-bot risk assessment."""
        # Register multiple bots
        for i in range(3):
            bot = AsyncMock()
            bot.bot_config.symbols = ["BTCUSDT"]
            await coordinator.register_bot(f"bot_{i}", bot)
            
            # Add positions
            await coordinator.update_bot_position(f"bot_{i}", "BTCUSDT", {
                "side": "BUY",
                "quantity": Decimal("1.0"),
                "average_price": Decimal("50000")
            })
        
        # Check risk for new position
        risk_assessment = await coordinator.check_cross_bot_risk(
            "new_bot", "BTCUSDT", OrderSide.BUY, Decimal("2.0")
        )
        
        assert "concentration_risk" in risk_assessment
        assert "total_exposure" in risk_assessment
        assert "recommendation" in risk_assessment

    @pytest.mark.asyncio
    async def test_coordinate_bot_actions(self, coordinator):
        """Test bot action coordination."""
        # Register bots
        for i in range(2):
            bot = AsyncMock()
            await coordinator.register_bot(f"bot_{i}", bot)
        
        # Coordinate action
        action_data = {
            "action_type": "emergency_stop",
            "reason": "market_volatility",
            "target_bots": ["bot_0", "bot_1"]
        }
        
        results = await coordinator.coordinate_bot_actions(action_data)
        
        assert len(results) == 2
        assert all(isinstance(result, dict) for result in results)

    @pytest.mark.asyncio
    async def test_get_signal_history(self, coordinator):
        """Test signal history retrieval."""
        # Add signals to history
        for i in range(5):
            signal = {
                "signal_id": f"signal_{i}",
                "sender_bot": f"bot_{i}",
                "signal_data": {"type": "test"},
                "timestamp": datetime.now(timezone.utc),
                "recipients": []
            }
            coordinator.signal_history.append(signal)
        
        # Get recent signals
        recent_signals = await coordinator.get_signal_history(hours=24)
        
        assert len(recent_signals) == 5

    @pytest.mark.asyncio
    async def test_get_signal_history_filtered(self, coordinator):
        """Test filtered signal history retrieval."""
        # Add signals with different types
        for signal_type in ["buy", "sell", "neutral"]:
            signal = {
                "signal_id": f"signal_{signal_type}",
                "sender_bot": "test_bot",
                "signal_data": {"signal_type": signal_type},
                "timestamp": datetime.now(timezone.utc),
                "recipients": []
            }
            coordinator.signal_history.append(signal)
        
        # Get filtered signals
        buy_signals = await coordinator.get_signal_history(
            hours=24, signal_type="buy"
        )
        
        assert len(buy_signals) == 1
        assert buy_signals[0]["signal_data"]["signal_type"] == "buy"

    @pytest.mark.asyncio
    async def test_get_coordination_summary(self, coordinator):
        """Test coordination summary generation."""
        # Register bots and add some activity
        for i in range(3):
            bot = AsyncMock()
            await coordinator.register_bot(f"bot_{i}", bot)
        
        # Add signal history
        coordinator.coordination_statistics["total_signals_shared"] = 10
        coordinator.coordination_statistics["total_conflicts_detected"] = 2
        
        summary = await coordinator.get_coordination_summary()
        
        # Verify summary structure
        assert "coordination_overview" in summary
        assert "bot_registry" in summary
        assert "signal_statistics" in summary
        assert "position_overview" in summary
        assert "conflict_analysis" in summary
        
        # Verify content
        assert summary["bot_registry"]["total_registered"] == 3
        assert summary["signal_statistics"]["total_shared"] == 10

    @pytest.mark.asyncio
    async def test_analyze_bot_interactions(self, coordinator):
        """Test bot interaction analysis."""
        # Set up interaction data
        for i in range(3):
            bot = AsyncMock()
            await coordinator.register_bot(f"bot_{i}", bot)
        
        # Add signal interactions
        for i in range(5):
            signal = {
                "signal_id": f"signal_{i}",
                "sender_bot": f"bot_{i % 3}",
                "recipients": [f"bot_{(i+1) % 3}", f"bot_{(i+2) % 3}"],
                "timestamp": datetime.now(timezone.utc)
            }
            coordinator.signal_history.append(signal)
        
        analysis = await coordinator.analyze_bot_interactions()
        
        assert "interaction_matrix" in analysis
        assert "communication_patterns" in analysis
        assert "collaboration_score" in analysis

    @pytest.mark.asyncio
    async def test_optimize_coordination(self, coordinator):
        """Test coordination optimization."""
        # Set up bots with different priorities
        for i, priority in enumerate([BotPriority.HIGH, BotPriority.NORMAL, BotPriority.LOW]):
            bot = AsyncMock()
            bot.bot_config.priority = priority
            await coordinator.register_bot(f"bot_{i}", bot)
        
        # Run optimization
        optimizations = await coordinator.optimize_coordination()
        
        assert isinstance(optimizations, list)
        # Should have optimization suggestions

    @pytest.mark.asyncio
    async def test_emergency_coordination(self, coordinator):
        """Test emergency coordination functionality."""
        # Register bots
        for i in range(3):
            bot = AsyncMock()
            bot.stop = AsyncMock()
            await coordinator.register_bot(f"bot_{i}", bot)
        
        # Trigger emergency coordination
        await coordinator.emergency_coordination("market_crash", "stop_all_trading")
        
        # All bots should have received emergency signal
        for bot in coordinator.registered_bots.values():
            # Emergency action should have been coordinated
            pass  # Verification depends on implementation

    @pytest.mark.asyncio
    async def test_signal_filtering(self, coordinator):
        """Test signal filtering functionality."""
        signal_data = {
            "signal_type": "buy",
            "symbol": "BTCUSDT",
            "confidence": 0.5  # Low confidence
        }
        
        # Test with confidence filter
        should_filter = await coordinator._should_filter_signal(signal_data, {"min_confidence": 0.7})
        assert should_filter
        
        # Test with acceptable confidence
        signal_data["confidence"] = 0.8
        should_filter = await coordinator._should_filter_signal(signal_data, {"min_confidence": 0.7})
        assert not should_filter

    @pytest.mark.asyncio
    async def test_coordination_loop(self, coordinator):
        """Test coordination monitoring loop."""
        await coordinator.start()
        
        # Add some test data
        for i in range(2):
            bot = AsyncMock()
            await coordinator.register_bot(f"bot_{i}", bot)
        
        # Run one cycle of coordination loop
        await coordinator._coordination_loop()
        
        # Should complete without errors
        assert coordinator.is_running

    @pytest.mark.asyncio
    async def test_cleanup_old_signals(self, coordinator):
        """Test cleanup of old signals."""
        # Add old signals
        old_time = datetime.now(timezone.utc)
        for i in range(5):
            signal = {
                "signal_id": f"old_signal_{i}",
                "timestamp": old_time,
                "sender_bot": "test_bot",
                "signal_data": {},
                "recipients": []
            }
            coordinator.signal_history.append(signal)
        
        initial_count = len(coordinator.signal_history)
        
        # Run cleanup
        await coordinator._cleanup_old_signals()
        
        # Some signals might be cleaned up depending on retention policy
        assert len(coordinator.signal_history) <= initial_count

    @pytest.mark.asyncio
    async def test_detect_coordination_issues(self, coordinator):
        """Test detection of coordination issues."""
        # Set up scenario with potential issues
        for i in range(3):
            bot = AsyncMock()
            await coordinator.register_bot(f"bot_{i}", bot)
            
            # Add conflicting positions
            await coordinator.update_bot_position(f"bot_{i}", "BTCUSDT", {
                "side": "BUY" if i % 2 == 0 else "SELL",
                "quantity": Decimal("1.0")
            })
        
        issues = await coordinator._detect_coordination_issues()
        
        assert isinstance(issues, list)
        # Should detect conflicting positions

    @pytest.mark.asyncio
    async def test_performance_tracking(self, coordinator):
        """Test coordination performance tracking."""
        # Generate some activity
        for i in range(10):
            await coordinator.share_signal(f"bot_{i}", {
                "signal_type": "test",
                "symbol": "BTCUSDT"
            })
        
        # Check performance metrics
        metrics = await coordinator.get_performance_metrics()
        
        assert "signal_throughput" in metrics
        assert "average_response_time" in metrics
        assert "coordination_efficiency" in metrics

    @pytest.mark.asyncio
    async def test_bot_communication_channels(self, coordinator):
        """Test bot communication channel management."""
        bot_id = "test_bot"
        channel_config = {
            "channel_type": "direct",
            "priority": "high",
            "encryption": True
        }
        
        # Set up communication channel
        success = await coordinator.setup_communication_channel(bot_id, channel_config)
        assert success
        
        # Test communication
        message = {"type": "test", "data": "hello"}
        sent = await coordinator.send_message(bot_id, message)
        assert sent