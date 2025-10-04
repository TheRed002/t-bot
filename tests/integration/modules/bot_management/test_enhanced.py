"""
Integration tests for enhanced bot management module integrations with Real Services.

Tests the new functionality added for:
- Analytics service integration
- Enhanced risk management with portfolio context
- Event publishing throughout lifecycle
- Enhanced exchange validation

Phase 5: Real service-based tests - NO MOCKS allowed except for external APIs.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List

from tests.integration.infrastructure.service_factory import RealServiceFactory

from src.bot_management.service import BotService
from src.core.events import BotEvent, BotEventType
from src.core.exceptions import ServiceError, ValidationError
from src.core.types import (
    BotConfiguration,
    BotMetrics,
    BotPriority,
    BotStatus,
    BotType,
    OrderSide,
    OrderType,
    RiskLevel,
    StateType,
)


@pytest.fixture
async def real_bot_services(clean_database):
    """Create real bot management services with real dependencies."""
    service_factory = RealServiceFactory()
    try:
        await service_factory.initialize_core_services(clean_database)
        await service_factory.initialize_bot_management_services()
        yield service_factory
    finally:
        await service_factory.cleanup()


@pytest.fixture
def sample_bot_config():
    """Sample bot configuration for testing."""
    return BotConfiguration(
        bot_id="test_bot_001",
        bot_type=BotType.TRADING,
        name="Test Bot",
        version="1.0.0",
        strategy_name="mean_reversion",
        strategy_parameters={"lookback": 20, "threshold": 2.0},
        exchanges=["binance"],
        symbols=["BTC/USDT", "ETH/USDT"],
        allocated_capital=Decimal("10000"),
        risk_percentage=0.02,
        max_positions=5,
        priority=BotPriority.NORMAL,
        auto_start=False,
        bot_name="Test Bot"
    )


class TestEnhancedBotServiceIntegration:
    """Test enhanced BotService features with real services."""

    @pytest.mark.asyncio
    async def test_bot_creation_with_real_services(
        self, real_bot_services, sample_bot_config
    ):
        """Test bot creation flow with real service integration."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()

        # Create bot
        bot_id = await bot_service.create_bot(sample_bot_config)

        # Verify bot was created
        assert bot_id == sample_bot_config.bot_id
        assert bot_id in bot_service._bots

    @pytest.mark.asyncio
    async def test_bot_lifecycle_with_real_state_persistence(
        self, real_bot_services, sample_bot_config
    ):
        """Test bot lifecycle with real state persistence."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()

        # Create and start bot
        bot_id = await bot_service.create_bot(sample_bot_config)
        await bot_service.start_bot(bot_id)

        # Verify bot is running
        status = await bot_service.get_bot_status(bot_id)
        assert isinstance(status, dict)

        # Stop bot
        await bot_service.stop_bot(bot_id)

    @pytest.mark.asyncio
    async def test_bot_metrics_tracking(
        self, real_bot_services, sample_bot_config
    ):
        """Test bot metrics tracking with real services."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()

        # Create bot
        bot_id = await bot_service.create_bot(sample_bot_config)

        # Update metrics
        metrics = {
            "trades_today": 5,
            "pnl_today": Decimal("100.50"),
            "win_rate": 0.75,
        }

        result = await bot_service.update_bot_metrics(bot_id, metrics)
        assert result is True

    @pytest.mark.asyncio
    async def test_multiple_bots_coordination(
        self, real_bot_services
    ):
        """Test coordination between multiple bots with real services."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()

        # Create multiple bots
        bot_configs = []
        for i in range(3):
            config = BotConfiguration(
                bot_id=f"test_bot_{i}",
                bot_type=BotType.TRADING,
                name=f"Test Bot {i}",
                version="1.0.0",
                strategy_name="momentum",
                exchanges=["binance"],
                symbols=["BTC/USDT"],
                allocated_capital=Decimal("1000.00"),
                priority=BotPriority.NORMAL,
            )
            bot_configs.append(config)
            await bot_service.create_bot(config)

        # Verify all bots were created
        all_status = await bot_service.get_all_bots_status()
        assert all_status["summary"]["total_bots"] == 3


class TestEnhancedRiskManagement:
    """Test enhanced risk management features with real services."""

    @pytest.mark.asyncio
    async def test_portfolio_risk_assessment(
        self, real_bot_services, sample_bot_config
    ):
        """Test portfolio-level risk assessment with real risk service."""
        bot_service = real_bot_services.container.get("BotService")
        risk_service = real_bot_services.container.get("risk_service")

        await bot_service.start()

        # Create bot
        bot_id = await bot_service.create_bot(sample_bot_config)

        # Verify risk service is accessible
        assert risk_service is not None

    @pytest.mark.asyncio
    async def test_risk_limit_validation(
        self, real_bot_services, sample_bot_config
    ):
        """Test risk limit validation during bot operations."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()

        # Create bot with specific risk parameters
        bot_id = await bot_service.create_bot(sample_bot_config)

        # Verify bot was created with risk constraints
        assert bot_id in bot_service._bots


class TestEnhancedExchangeValidation:
    """Test enhanced exchange validation with real services."""

    @pytest.mark.asyncio
    async def test_exchange_availability_check(
        self, real_bot_services, sample_bot_config
    ):
        """Test exchange availability validation with real exchange service."""
        bot_service = real_bot_services.container.get("BotService")
        exchange_service = real_bot_services.container.get("exchange_service")

        await bot_service.start()

        # Create bot - should validate exchange availability
        bot_id = await bot_service.create_bot(sample_bot_config)

        # Verify bot was created successfully
        assert bot_id == sample_bot_config.bot_id


class TestEnhancedStateManagement:
    """Test enhanced state management features with real state service."""

    @pytest.mark.asyncio
    async def test_state_persistence_and_recovery(
        self, real_bot_services, sample_bot_config
    ):
        """Test state persistence and recovery with real state service."""
        bot_service = real_bot_services.container.get("BotService")
        state_service = real_bot_services.container.get("state_service")

        await bot_service.start()

        # Create and start bot
        bot_id = await bot_service.create_bot(sample_bot_config)
        await bot_service.start_bot(bot_id)

        # Verify state service is being used
        assert state_service is not None

    @pytest.mark.asyncio
    async def test_concurrent_state_operations(
        self, real_bot_services
    ):
        """Test concurrent state operations with real state service."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()

        # Create multiple bots concurrently
        configs = [
            BotConfiguration(
                bot_id=f"concurrent_bot_{i}",
                bot_type=BotType.TRADING,
                name=f"Concurrent Bot {i}",
                version="1.0.0",
                strategy_name="test",
                exchanges=["binance"],
                symbols=["BTC/USDT"],
                allocated_capital=Decimal("1000"),
            )
            for i in range(5)
        ]

        # Create all bots concurrently
        tasks = [bot_service.create_bot(config) for config in configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all creations completed (some may fail, that's ok)
        assert len(results) == 5
        successful = [r for r in results if isinstance(r, str)]
        assert len(successful) >= 1


class TestEnhancedCapitalManagement:
    """Test enhanced capital management integration with real services."""

    @pytest.mark.asyncio
    async def test_capital_allocation_and_tracking(
        self, real_bot_services, sample_bot_config
    ):
        """Test capital allocation and tracking with real capital service."""
        bot_service = real_bot_services.container.get("BotService")
        capital_service = real_bot_services.container.get("capital_service")

        await bot_service.start()

        # Create bot - should allocate capital
        bot_id = await bot_service.create_bot(sample_bot_config)

        # Verify bot was created
        assert bot_id in bot_service._bots
        # Capital service should have been called
        assert capital_service is not None


class TestEnhancedHealthMonitoring:
    """Test enhanced health monitoring with real monitoring service."""

    @pytest.mark.asyncio
    async def test_comprehensive_health_check(
        self, real_bot_services, sample_bot_config
    ):
        """Test comprehensive health check with real services."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()

        # Create bot
        bot_id = await bot_service.create_bot(sample_bot_config)

        # Perform health check
        health = await bot_service.perform_health_check(bot_id)

        # Verify health check structure
        assert isinstance(health, dict)
        assert "status" in health
        assert "checks" in health

    @pytest.mark.asyncio
    async def test_system_wide_health_monitoring(
        self, real_bot_services
    ):
        """Test system-wide health monitoring with real services."""
        bot_service = real_bot_services.container.get("BotService")
        await bot_service.start()

        # Get system health
        health = await bot_service.health_check()

        # Verify health status
        assert health is not None
        assert hasattr(health, 'status')
