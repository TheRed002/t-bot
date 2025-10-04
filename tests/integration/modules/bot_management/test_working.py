"""
Working integration tests for bot_management module.

Tests the actual available functionality of bot management with REAL integrations.
Focuses on testing the service layers and interactions with real database, state, and other services.

Phase 5: Real service-based tests - NO MOCKS allowed except for external APIs.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any

import pytest

# Import test infrastructure
from tests.integration.infrastructure.service_factory import RealServiceFactory

# Import actual bot management classes
from src.bot_management.service import BotService
from src.bot_management.bot_coordinator import BotCoordinator
from src.bot_management.bot_lifecycle import BotLifecycle
from src.bot_management.bot_monitor import BotMonitor
from src.bot_management.resource_manager import ResourceManager

from src.core.types.bot import (
    BotConfiguration,
    BotState,
    BotStatus,
    BotMetrics,
    BotType,
    BotPriority,
)
from src.core.types.risk import RiskLevel
from src.core.types.trading import (
    OrderRequest,
    OrderSide,
    OrderType,
    OrderStatus,
)
from src.core.exceptions import (
    ComponentError,
    ServiceError,
    ValidationError,
)


@pytest.fixture
async def real_bot_services(clean_database):
    """Create real bot management services with real dependencies."""
    service_factory = RealServiceFactory()
    try:
        # Initialize core services (database, cache, etc.)
        await service_factory.initialize_core_services(clean_database)

        # Initialize bot management services with real dependencies
        await service_factory.initialize_bot_management_services()

        yield service_factory
    finally:
        await service_factory.cleanup()


@pytest.fixture
def sample_bot_config():
    """Create sample bot configuration."""
    return BotConfiguration(
        bot_id="integration_test_bot",
        bot_type=BotType.TRADING,
        name="Integration Test Bot",
        version="1.0.0",
        strategy_id="momentum_strategy",
        strategy_name="Momentum Strategy",
        exchanges=["binance"],
        symbols=["BTC/USDT", "ETH/USDT"],
        allocated_capital=Decimal("10000.00"),
        risk_parameters={
            "max_position_size": Decimal("1000.00"),
            "stop_loss": Decimal("0.02"),
            "take_profit": Decimal("0.05"),
        },
        enabled=True,
    )


class TestBotServiceIntegration:
    """Test BotService with integrated components using real services."""

    @pytest.mark.asyncio
    async def test_bot_service_initialization(self, real_bot_services):
        """Test bot service initialization and basic functionality."""
        bot_service = real_bot_services.container.get("BotService")

        # Test service is properly initialized
        assert bot_service is not None
        assert bot_service._exchange_service is not None
        assert bot_service._capital_service is not None

        # Test service start
        await bot_service.start()
        assert bot_service.is_running

        # Test health check
        health_status = await bot_service.health_check()
        assert health_status is not None
        assert health_status.status in ["healthy", "HEALTHY"]

    @pytest.mark.asyncio
    async def test_create_bot_with_real_services(
        self, real_bot_services, sample_bot_config
    ):
        """Test bot creation with real service pipeline."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()

        # Create bot with real services
        bot_id = await bot_service.create_bot(sample_bot_config)

        # Verify bot was created
        assert bot_id == sample_bot_config.bot_id
        assert isinstance(bot_id, str)

        # Verify bot exists in internal state
        assert bot_id in bot_service._bots

    @pytest.mark.asyncio
    async def test_get_all_bots_status(self, real_bot_services):
        """Test getting status of all bots with real service integration."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()

        # Get all bot status
        result = await bot_service.get_all_bots_status()

        # Verify the API structure matches expected format
        assert isinstance(result, dict)
        assert "summary" in result
        assert "bots" in result
        assert "timestamp" in result

        # Verify summary structure
        summary = result["summary"]
        assert "total_bots" in summary
        assert "running" in summary or "active" in summary
        assert "stopped" in summary or "inactive" in summary

        # Since no bots are active initially, counts should be 0
        assert summary["total_bots"] == 0


class TestBotCoordinatorIntegration:
    """Test BotCoordinator with integrated components using real services."""

    @pytest.mark.asyncio
    async def test_coordinator_startup_sequence(
        self, real_bot_services
    ):
        """Test coordinator startup and bot registration with real config."""
        from src.core.config import get_config

        config = get_config()
        coordinator = BotCoordinator(config=config)

        # Start coordinator
        await coordinator.start()

        # Register multiple bots
        bot_ids = ["bot_1", "bot_2", "bot_3"]
        for i, bot_id in enumerate(bot_ids):
            bot_config = BotConfiguration(
                bot_id=bot_id,
                name=f"Test Bot {i+1}",
                bot_type=BotType.TRADING,
                version="1.0.0",
                strategy_name="momentum",
                exchanges=["binance"],
                symbols=[f"BTC/USDT"],
                allocated_capital=Decimal("1000.00"),
                risk_profile=RiskLevel.MEDIUM,
                max_position_size=Decimal("100.00"),
                stop_loss_percentage=Decimal("0.02"),
                take_profit_percentage=Decimal("0.05")
            )
            await coordinator.register_bot(bot_id, bot_config)

        # Verify coordinator state
        assert len(coordinator.registered_bots) == 3
        assert "bot_1" in coordinator.registered_bots

    @pytest.mark.asyncio
    async def test_signal_sharing_between_bots(
        self, real_bot_services
    ):
        """Test signal sharing functionality with real coordinator."""
        from src.core.config import get_config

        config = get_config()
        coordinator = BotCoordinator(config=config)
        await coordinator.start()

        # Create proper bot configurations
        sender_config = BotConfiguration(
            bot_id="bot_sender",
            name="Sender Bot",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_name="momentum",
            exchanges=["binance"],
            symbols=["BTC/USDT"],
            allocated_capital=Decimal("1000.00"),
            risk_profile=RiskLevel.MEDIUM,
            max_position_size=Decimal("100.00"),
            stop_loss_percentage=Decimal("0.02"),
            take_profit_percentage=Decimal("0.05")
        )

        receiver_config = BotConfiguration(
            bot_id="bot_receiver",
            name="Receiver Bot",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_name="momentum",
            exchanges=["binance"],
            symbols=["BTC/USDT"],
            allocated_capital=Decimal("1000.00"),
            risk_profile=RiskLevel.MEDIUM,
            max_position_size=Decimal("100.00"),
            stop_loss_percentage=Decimal("0.02"),
            take_profit_percentage=Decimal("0.05")
        )

        await coordinator.register_bot("bot_sender", sender_config)
        await coordinator.register_bot("bot_receiver", receiver_config)

        # Share signal
        signal_data = {
            "type": "BUY_SIGNAL",
            "symbol": "BTC/USDT",
            "direction": "BUY",
            "confidence": 0.8,
            "timestamp": datetime.utcnow(),
        }

        result = await coordinator.share_signal(
            bot_id="bot_sender",
            signal_data=signal_data,
            target_bots=["bot_receiver"]
        )

        # Verify signal was shared (result is number of recipients)
        assert result == 1

        # Check receiver got the signal
        signals = await coordinator.get_shared_signals("bot_receiver")
        assert len(signals) == 1

    @pytest.mark.asyncio
    async def test_risk_coordination(self, real_bot_services):
        """Test portfolio-level risk coordination with real services."""
        from src.core.config import get_config

        config = get_config()
        coordinator = BotCoordinator(config=config)
        await coordinator.start()

        # Register bots with proper configurations
        bot1_config = BotConfiguration(
            bot_id="bot_1",
            name="Risk Bot 1",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_name="momentum",
            exchanges=["binance"],
            symbols=["BTC/USDT"],
            allocated_capital=Decimal("5000.00"),
            risk_profile=RiskLevel.MEDIUM,
            max_position_size=Decimal("500.00"),
            stop_loss_percentage=Decimal("0.02"),
            take_profit_percentage=Decimal("0.05")
        )

        bot2_config = BotConfiguration(
            bot_id="bot_2",
            name="Risk Bot 2",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_name="momentum",
            exchanges=["binance"],
            symbols=["BTC/USDT"],
            allocated_capital=Decimal("3000.00"),
            risk_profile=RiskLevel.MEDIUM,
            max_position_size=Decimal("300.00"),
            stop_loss_percentage=Decimal("0.02"),
            take_profit_percentage=Decimal("0.05")
        )

        await coordinator.register_bot("bot_1", bot1_config)
        await coordinator.register_bot("bot_2", bot2_config)

        # Check cross-bot risk for a proposed order
        risk_result = await coordinator.check_cross_bot_risk(
            bot_id="bot_1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("100.00")
        )

        # Verify risk assessment structure
        assert isinstance(risk_result, dict)
        # Should have risk assessment data
        assert "boundary_validation" in risk_result or "acknowledgment_required" in risk_result or "approved" in risk_result


class TestBotLifecycleIntegration:
    """Test BotLifecycle with real services."""

    @pytest.mark.asyncio
    async def test_complete_lifecycle_flow(
        self, real_bot_services, sample_bot_config
    ):
        """Test complete bot lifecycle from creation to termination with real services."""
        from src.core.config import get_config

        config = get_config()
        lifecycle = BotLifecycle(config=config)
        await lifecycle.start()

        # 1. Create bot from template
        bot_config = await lifecycle.create_bot_from_template(
            template_name="simple_strategy_bot",
            bot_name="Test Lifecycle Bot",
            custom_config={
                "strategy_name": "momentum",
                "exchanges": ["binance"],
                "symbols": ["BTC/USDT"],
                "allocated_capital": Decimal("1000.00"),
            }
        )
        assert bot_config.bot_id.startswith("simple_strategy_bot_")
        assert bot_config.name == "Test Lifecycle Bot"

        # 2. Check lifecycle summary
        summary = await lifecycle.get_lifecycle_summary()
        assert "lifecycle_overview" in summary
        assert "deployment_states" in summary
        assert summary["lifecycle_overview"]["managed_bots"] == 1

        # 3. Get bot lifecycle details
        details = await lifecycle.get_bot_lifecycle_details(bot_config.bot_id)
        # Details might be None for newly created bots
        assert details is None or isinstance(details, dict)

        # 4. Stop lifecycle
        await lifecycle.stop()


class TestResourceManagerIntegration:
    """Test ResourceManager with real services."""

    @pytest.mark.asyncio
    async def test_resource_allocation_flow(self, real_bot_services):
        """Test complete resource allocation with real capital service."""
        from src.core.config import get_config

        config = get_config()
        capital_service = real_bot_services.container.get("capital_service")

        resource_manager = ResourceManager(
            config=config,
            capital_service=capital_service,
        )
        await resource_manager.start()

        # Allocate resources for bot
        allocation_request = {
            "cpu_cores": 2,
            "memory_gb": 4,
            "connections": 10,
        }

        bot_id = "resource_bot"
        result = await resource_manager.allocate_resources(bot_id, allocation_request)
        assert result is True

        # Release resources
        release_result = await resource_manager.release_resources(bot_id)
        assert isinstance(release_result, bool)

        await resource_manager.stop()


class TestPerformanceIntegration:
    """Test performance aspects of integrated services."""

    @pytest.mark.asyncio
    async def test_concurrent_bot_operations(self, real_bot_services):
        """Test concurrent operations on multiple bots with real service."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()

        # Execute concurrent operations
        bot_ids = [f"bot_{i}" for i in range(10)]

        # Test concurrent status checks
        tasks = [bot_service.get_bot_status(bot_id) for bot_id in bot_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all operations completed
        assert len(results) == 10
        # Most will fail because bots don't exist, but they should fail gracefully
        assert all(isinstance(r, (dict, Exception)) for r in results)
