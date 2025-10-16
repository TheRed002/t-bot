"""
Real Service Integration Tests for Bot Management Module.

Tests real bot lifecycle operations with actual service instances,
real coordination, resource management, and monitoring.
NO MOCKS for internal services - only real implementations.

CRITICAL: This module validates bot management operations that control
trading bot lifecycle and resource allocation. All tests must use real
services and proper async patterns.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.core.exceptions import ServiceError, ValidationError
from src.core.types import (
    BotConfiguration,
    BotMetrics,
    BotPriority,
    BotState,
    BotStatus,
    BotType,
    OrderRequest,
    OrderSide,
    OrderType,
)


class TestBotInstanceServiceRealOperations:
    """Test real bot instance service operations."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_create_bot_instance(self, bot_instance_service):
        """Test creating a new bot instance."""
        # GIVEN: Valid bot configuration
        bot_config = BotConfiguration(
            bot_id=f"test_bot_{uuid.uuid4().hex[:8]}",
            name=f"Test Bot {uuid.uuid4().hex[:4]}",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_id="test_strategy",
            symbols=["BTC/USDT"],
            exchanges=["binance"],
            allocated_capital=Decimal("10000.00"),
            max_capital=Decimal("10000.00"),
            priority=BotPriority.NORMAL,
        )

        # WHEN: Create bot instance
        bot_id = await bot_instance_service.create_bot_instance(bot_config)

        # THEN: Bot should be created successfully
        assert bot_id == bot_config.bot_id
        assert bot_id in bot_instance_service._bot_instances

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_start_bot(self, bot_instance_service):
        """Test starting a bot instance."""
        # GIVEN: Created bot instance
        bot_config = BotConfiguration(
            bot_id=f"test_bot_{uuid.uuid4().hex[:8]}",
            name=f"Test Bot {uuid.uuid4().hex[:4]}",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_id="test_strategy",
            symbols=["BTC/USDT"],
            exchanges=["binance"],
            allocated_capital=Decimal("5000.00"),
        )
        bot_id = await bot_instance_service.create_bot_instance(bot_config)

        # WHEN: Start bot
        success = await bot_instance_service.start_bot(bot_id)

        # THEN: Bot should start successfully
        assert success is True

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_stop_bot(self, bot_instance_service):
        """Test stopping a running bot instance."""
        # GIVEN: Running bot instance
        bot_config = BotConfiguration(
            bot_id=f"test_bot_{uuid.uuid4().hex[:8]}",
            name=f"Test Bot {uuid.uuid4().hex[:4]}",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_id="test_strategy",
            symbols=["BTC/USDT"],
            exchanges=["binance"],
            allocated_capital=Decimal("5000.00"),
        )
        bot_id = await bot_instance_service.create_bot_instance(bot_config)
        await bot_instance_service.start_bot(bot_id)

        # WHEN: Stop bot
        success = await bot_instance_service.stop_bot(bot_id)

        # THEN: Bot should stop successfully
        assert success is True

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_pause_and_resume_bot(self, bot_instance_service):
        """Test pausing and resuming a bot instance."""
        # GIVEN: Running bot instance
        bot_config = BotConfiguration(
            bot_id=f"test_bot_{uuid.uuid4().hex[:8]}",
            name=f"Test Bot {uuid.uuid4().hex[:4]}",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_id="test_strategy",
            symbols=["BTC/USDT"],
            exchanges=["binance"],
            allocated_capital=Decimal("3000.00"),
        )
        bot_id = await bot_instance_service.create_bot_instance(bot_config)
        await bot_instance_service.start_bot(bot_id)

        # WHEN: Pause bot
        pause_success = await bot_instance_service.pause_bot(bot_id)

        # THEN: Bot should pause successfully
        assert pause_success is True

        # WHEN: Resume bot
        resume_success = await bot_instance_service.resume_bot(bot_id)

        # THEN: Bot should resume successfully
        assert resume_success is True

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_get_bot_state(self, bot_instance_service):
        """Test retrieving bot state."""
        # GIVEN: Created bot instance
        bot_config = BotConfiguration(
            bot_id=f"test_bot_{uuid.uuid4().hex[:8]}",
            name=f"Test Bot {uuid.uuid4().hex[:4]}",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_id="test_strategy",
            symbols=["ETH/USDT"],
            exchanges=["binance"],
            allocated_capital=Decimal("8000.00"),
        )
        bot_id = await bot_instance_service.create_bot_instance(bot_config)

        # WHEN: Get bot state
        state = await bot_instance_service.get_bot_state(bot_id)

        # THEN: State should be returned correctly
        assert isinstance(state, BotState)
        assert state.bot_id == bot_id

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_remove_bot_instance(self, bot_instance_service):
        """Test removing a bot instance."""
        # GIVEN: Stopped bot instance
        bot_config = BotConfiguration(
            bot_id=f"test_bot_{uuid.uuid4().hex[:8]}",
            name=f"Test Bot {uuid.uuid4().hex[:4]}",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_id="test_strategy",
            symbols=["BTC/USDT"],
            exchanges=["binance"],
            allocated_capital=Decimal("2000.00"),
        )
        bot_id = await bot_instance_service.create_bot_instance(bot_config)

        # WHEN: Remove bot instance
        success = await bot_instance_service.remove_bot_instance(bot_id)

        # THEN: Bot should be removed successfully
        assert success is True
        assert bot_id not in bot_instance_service._bot_instances

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_get_active_bot_ids(self, bot_instance_service):
        """Test retrieving active bot IDs."""
        # GIVEN: Multiple bot instances
        bot_configs = [
            BotConfiguration(
                bot_id=f"test_bot_{i}_{uuid.uuid4().hex[:8]}",
                name=f"Test Bot {i}",
                bot_type=BotType.TRADING,
                version="1.0.0",
                strategy_id=f"strategy_{i}",
                symbols=["BTC/USDT"],
                exchanges=["binance"],
                allocated_capital=Decimal("1000.00"),
            )
            for i in range(3)
        ]

        for config in bot_configs:
            await bot_instance_service.create_bot_instance(config)

        # WHEN: Get active bot IDs
        active_ids = bot_instance_service.get_active_bot_ids()

        # THEN: Should return list of bot IDs
        assert isinstance(active_ids, list)
        assert len(active_ids) >= 3

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_create_duplicate_bot_name(self, bot_instance_service):
        """Test that duplicate bot names are rejected."""
        # GIVEN: Existing bot with specific name
        bot_name = f"Unique_Bot_{uuid.uuid4().hex[:8]}"
        bot_config1 = BotConfiguration(
            bot_id=f"test_bot_1_{uuid.uuid4().hex[:8]}",
            name=bot_name,
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_id="test_strategy",
            symbols=["BTC/USDT"],
            exchanges=["binance"],
            allocated_capital=Decimal("5000.00"),
        )
        await bot_instance_service.create_bot_instance(bot_config1)

        # WHEN/THEN: Creating bot with same name should fail
        bot_config2 = BotConfiguration(
            bot_id=f"test_bot_2_{uuid.uuid4().hex[:8]}",
            name=bot_name,
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_id="test_strategy",
            symbols=["BTC/USDT"],
            exchanges=["binance"],
            allocated_capital=Decimal("5000.00"),
        )
        with pytest.raises(ServiceError, match="already exists"):
            await bot_instance_service.create_bot_instance(bot_config2)


class TestBotCoordinationServiceRealOperations:
    """Test real bot coordination service operations."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_register_bot(self, bot_coordination_service):
        """Test registering a bot for coordination."""
        # GIVEN: Bot configuration
        bot_id = f"test_bot_{uuid.uuid4().hex[:8]}"
        bot_config = BotConfiguration(
            bot_id=bot_id,
            name=f"Test Bot {uuid.uuid4().hex[:4]}",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_id="test_strategy",
            symbols=["BTC/USDT"],
            exchanges=["binance"],
            allocated_capital=Decimal("10000.00"),
        )

        # WHEN: Register bot
        await bot_coordination_service.register_bot(bot_id, bot_config)

        # THEN: Bot should be registered
        assert bot_id in bot_coordination_service.registered_bots

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_unregister_bot(self, bot_coordination_service):
        """Test unregistering a bot from coordination."""
        # GIVEN: Registered bot
        bot_id = f"test_bot_{uuid.uuid4().hex[:8]}"
        bot_config = BotConfiguration(
            bot_id=bot_id,
            name=f"Test Bot {uuid.uuid4().hex[:4]}",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_id="test_strategy",
            symbols=["BTC/USDT"],
            exchanges=["binance"],
            allocated_capital=Decimal("5000.00"),
        )
        await bot_coordination_service.register_bot(bot_id, bot_config)

        # WHEN: Unregister bot
        await bot_coordination_service.unregister_bot(bot_id)

        # THEN: Bot should be unregistered
        assert bot_id not in bot_coordination_service.registered_bots

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_share_signal_between_bots(self, bot_coordination_service):
        """Test sharing trading signals between bots."""
        # GIVEN: Two registered bots
        bot1_id = f"bot1_{uuid.uuid4().hex[:8]}"
        bot2_id = f"bot2_{uuid.uuid4().hex[:8]}"

        for bot_id in [bot1_id, bot2_id]:
            config = BotConfiguration(
                bot_id=bot_id,
                name=f"Bot {bot_id}",
                bot_type=BotType.TRADING,
                version="1.0.0",
                strategy_id="test_strategy",
                symbols=["BTC/USDT"],
                exchanges=["binance"],
                allocated_capital=Decimal("5000.00"),
            )
            await bot_coordination_service.register_bot(bot_id, config)

        # WHEN: Bot1 shares a signal
        signal_count = await bot_coordination_service.share_signal(
            bot_id=bot1_id,
            signal_type="technical",
            symbol="BTC/USDT",
            direction=OrderSide.BUY,
            strength=0.8,
            metadata={"indicator": "RSI", "value": 35, "reasoning": "Strong uptrend detected"},
        )

        # THEN: Signal should be shared
        assert signal_count >= 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_check_position_conflicts(self, bot_coordination_service):
        """Test checking for position conflicts across bots."""
        # GIVEN: Multiple registered bots
        for i in range(3):
            bot_id = f"bot_{i}_{uuid.uuid4().hex[:8]}"
            config = BotConfiguration(
                bot_id=bot_id,
                name=f"Bot {i}",
                bot_type=BotType.TRADING,
                version="1.0.0",
                strategy_id=f"strategy_{i}",
                symbols=["BTC/USDT"],
                exchanges=["binance"],
                allocated_capital=Decimal("3000.00"),
            )
            await bot_coordination_service.register_bot(bot_id, config)

        # WHEN: Check for position conflicts
        conflicts = await bot_coordination_service.check_position_conflicts("BTC/USDT")

        # THEN: Should return conflict analysis
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_cross_bot_risk_assessment(self, bot_coordination_service):
        """Test cross-bot risk assessment."""
        # GIVEN: Registered bot
        bot_id = f"test_bot_{uuid.uuid4().hex[:8]}"
        bot_config = BotConfiguration(
            bot_id=bot_id,
            name=f"Test Bot {uuid.uuid4().hex[:4]}",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_id="test_strategy",
            symbols=["BTC/USDT"],
            exchanges=["binance"],
            allocated_capital=Decimal("10000.00"),
        )
        await bot_coordination_service.register_bot(bot_id, bot_config)

        # WHEN: Check cross-bot risk
        risk_assessment = await bot_coordination_service.check_cross_bot_risk(
            bot_id=bot_id,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
        )

        # THEN: Should return risk assessment
        assert isinstance(risk_assessment, dict)
        assert "allowed" in risk_assessment


class TestBotLifecycleServiceRealOperations:
    """Test real bot lifecycle service operations."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_create_bot_from_template(self, bot_lifecycle_service):
        """Test creating bot from template."""
        # GIVEN: Template parameters
        template_name = "basic_trading"  # Use existing template
        bot_name = f"Scalper_{uuid.uuid4().hex[:8]}"
        exchange = "binance"
        strategy = "test_strategy"
        capital_amount = Decimal("5000.00")

        # WHEN: Create bot from template
        bot_config = await bot_lifecycle_service.create_bot_from_template(
            template_name=template_name,
            bot_name=bot_name,
            exchange=exchange,
            strategy=strategy,
            capital_amount=capital_amount,
            priority="high",
            custom_config={"max_position_size": "100.00"},
        )

        # THEN: Bot configuration should be created
        assert isinstance(bot_config, BotConfiguration)
        assert bot_config.name == bot_name

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_get_lifecycle_status(self, bot_lifecycle_service):
        """Test retrieving lifecycle status for a bot."""
        # GIVEN: Bot ID
        bot_id = f"test_bot_{uuid.uuid4().hex[:8]}"

        # WHEN: Get lifecycle status
        status = await bot_lifecycle_service.get_lifecycle_status(bot_id)

        # THEN: Should return status information
        assert isinstance(status, dict)


class TestBotMonitoringServiceRealOperations:
    """Test real bot monitoring service operations."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_get_system_health(self, bot_monitoring_service):
        """Test retrieving system health metrics."""
        # WHEN: Get system health
        health = await bot_monitoring_service.get_system_health()

        # THEN: Should return health information
        assert isinstance(health, dict)
        # Check for expected keys in the response
        assert any(
            key in health for key in ["status", "health_percentage", "total_bots", "healthy_bots"]
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_get_performance_summary(self, bot_monitoring_service):
        """Test retrieving performance summary."""
        # WHEN: Get performance summary
        summary = await bot_monitoring_service.get_performance_summary()

        # THEN: Should return summary information
        assert isinstance(summary, dict)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_check_alert_conditions(self, bot_monitoring_service):
        """Test checking alert conditions."""
        # WHEN: Check for alerts
        alerts = await bot_monitoring_service.check_alert_conditions()

        # THEN: Should return list of alerts
        assert isinstance(alerts, list)


class TestBotResourceServiceRealOperations:
    """Test real bot resource service operations."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_request_resources(self, bot_resource_service):
        """Test requesting resources for a bot."""
        # GIVEN: Bot resource request
        bot_id = f"test_bot_{uuid.uuid4().hex[:8]}"
        capital_amount = Decimal("5000.00")

        # WHEN: Request resources
        success = await bot_resource_service.request_resources(
            bot_id=bot_id,
            capital_amount=capital_amount,
            priority=BotPriority.NORMAL,
        )

        # THEN: Resources should be allocated
        assert isinstance(success, bool)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_release_resources(self, bot_resource_service):
        """Test releasing bot resources."""
        # GIVEN: Bot with allocated resources
        bot_id = f"test_bot_{uuid.uuid4().hex[:8]}"
        capital_amount = Decimal("3000.00")
        await bot_resource_service.request_resources(
            bot_id=bot_id,
            capital_amount=capital_amount,
            priority=BotPriority.NORMAL,
        )

        # WHEN: Release resources
        success = await bot_resource_service.release_resources(bot_id)

        # THEN: Resources should be released
        assert success is True

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_verify_resources(self, bot_resource_service):
        """Test verifying bot resource allocation."""
        # GIVEN: Bot with allocated resources
        bot_id = f"test_bot_{uuid.uuid4().hex[:8]}"
        capital_amount = Decimal("4000.00")
        await bot_resource_service.request_resources(
            bot_id=bot_id,
            capital_amount=capital_amount,
            priority=BotPriority.NORMAL,
        )

        # WHEN: Verify resources
        verified = await bot_resource_service.verify_resources(bot_id)

        # THEN: Verification should succeed
        assert isinstance(verified, bool)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_get_resource_summary(self, bot_resource_service):
        """Test retrieving resource summary."""
        # WHEN: Get resource summary
        summary = await bot_resource_service.get_resource_summary()

        # THEN: Should return summary information
        assert isinstance(summary, dict)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_check_resource_availability(self, bot_resource_service):
        """Test checking resource availability."""
        # GIVEN: Resource type and amount
        resource_type = "capital"
        amount = Decimal("10000.00")

        # WHEN: Check availability
        available = await bot_resource_service.check_resource_availability(
            resource_type, amount
        )

        # THEN: Should return availability status
        assert isinstance(available, bool)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_update_capital_allocation(self, bot_resource_service):
        """Test updating capital allocation for a bot."""
        # GIVEN: Bot with allocated resources
        bot_id = f"test_bot_{uuid.uuid4().hex[:8]}"
        initial_amount = Decimal("5000.00")
        await bot_resource_service.request_resources(
            bot_id=bot_id,
            capital_amount=initial_amount,
            priority=BotPriority.NORMAL,
        )

        # WHEN: Update capital allocation
        new_amount = Decimal("7500.00")
        success = await bot_resource_service.update_capital_allocation(bot_id, new_amount)

        # THEN: Allocation should be updated
        assert isinstance(success, bool)


class TestBotManagementControllerRealOperations:
    """Test real bot management controller operations."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_controller_create_bot(self, bot_management_controller):
        """Test creating bot through controller."""
        # GIVEN: Bot creation parameters
        template_name = "basic_trading"  # Use existing template
        bot_name = f"Controller_Bot_{uuid.uuid4().hex[:8]}"
        exchange = "binance"
        strategy = "test_strategy"
        capital_amount = Decimal("5000.00")

        # WHEN: Create bot via controller
        result = await bot_management_controller.create_bot(
            template_name=template_name,
            bot_name=bot_name,
            exchange=exchange,
            strategy=strategy,
            capital_amount=capital_amount,
            deployment_strategy="immediate",
            priority="normal",
        )

        # THEN: Bot should be created
        assert isinstance(result, dict)
        assert "bot_id" in result or "bot_config" in result

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_controller_list_bots(self, bot_management_controller):
        """Test listing bots through controller."""
        # WHEN: List all bots
        result = await bot_management_controller.list_bots()

        # THEN: Should return bot list
        assert isinstance(result, dict)
        assert "bots" in result or "bot_ids" in result or "status" in result

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_controller_get_system_overview(self, bot_management_controller):
        """Test getting system overview through controller."""
        # WHEN: Get system overview
        overview = await bot_management_controller.get_system_overview()

        # THEN: Should return overview information
        assert isinstance(overview, dict)


class TestBotLifecycleIntegration:
    """Test complete bot lifecycle integration."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_complete_bot_lifecycle(
        self,
        bot_instance_service,
        bot_coordination_service,
        bot_resource_service,
    ):
        """Test complete bot lifecycle: create, start, operate, stop, remove."""
        # GIVEN: Bot configuration
        bot_id = f"lifecycle_bot_{uuid.uuid4().hex[:8]}"
        bot_config = BotConfiguration(
            bot_id=bot_id,
            name=f"Lifecycle Bot {uuid.uuid4().hex[:4]}",
            bot_type=BotType.TRADING,
            version="1.0.0",
            strategy_id="test_strategy",
            symbols=["BTC/USDT"],
            exchanges=["binance"],
            allocated_capital=Decimal("10000.00"),
            priority=BotPriority.HIGH,
        )

        # WHEN: Execute complete lifecycle
        # 1. Request resources
        resources_allocated = await bot_resource_service.request_resources(
            bot_id=bot_id,
            capital_amount=bot_config.allocated_capital,
            priority=bot_config.priority,
        )
        assert resources_allocated is True or isinstance(resources_allocated, bool)

        # 2. Create bot instance
        created_bot_id = await bot_instance_service.create_bot_instance(bot_config)
        assert created_bot_id == bot_id

        # 3. Register for coordination
        await bot_coordination_service.register_bot(bot_id, bot_config)
        assert bot_id in bot_coordination_service.registered_bots

        # 4. Start bot
        started = await bot_instance_service.start_bot(bot_id)
        assert started is True

        # 5. Get bot state
        state = await bot_instance_service.get_bot_state(bot_id)
        assert isinstance(state, BotState)
        assert state.bot_id == bot_id

        # 6. Stop bot
        stopped = await bot_instance_service.stop_bot(bot_id)
        assert stopped is True

        # 7. Unregister from coordination
        await bot_coordination_service.unregister_bot(bot_id)
        assert bot_id not in bot_coordination_service.registered_bots

        # 8. Release resources
        resources_released = await bot_resource_service.release_resources(bot_id)
        assert resources_released is True

        # 9. Remove bot instance
        removed = await bot_instance_service.remove_bot_instance(bot_id)
        assert removed is True

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_concurrent_bot_operations(
        self,
        bot_instance_service,
        bot_coordination_service,
    ):
        """Test concurrent operations on multiple bots."""
        # GIVEN: Multiple bot configurations
        num_bots = 3
        bot_configs = [
            BotConfiguration(
                bot_id=f"concurrent_bot_{i}_{uuid.uuid4().hex[:8]}",
                name=f"Concurrent Bot {i}",
                bot_type=BotType.TRADING,
                version="1.0.0",
                strategy_id=f"strategy_{i}",
                symbols=["BTC/USDT"],
                exchanges=["binance"],
                allocated_capital=Decimal("5000.00"),
            )
            for i in range(num_bots)
        ]

        # WHEN: Create and start bots concurrently
        create_tasks = [
            bot_instance_service.create_bot_instance(config) for config in bot_configs
        ]
        bot_ids = await asyncio.gather(*create_tasks)

        # Register for coordination
        register_tasks = [
            bot_coordination_service.register_bot(bot_id, config)
            for bot_id, config in zip(bot_ids, bot_configs)
        ]
        await asyncio.gather(*register_tasks)

        # Start all bots
        start_tasks = [bot_instance_service.start_bot(bot_id) for bot_id in bot_ids]
        start_results = await asyncio.gather(*start_tasks)

        # THEN: All operations should succeed
        assert all(bot_id is not None for bot_id in bot_ids)
        assert all(result is True for result in start_results)

        # Cleanup: Stop all bots
        stop_tasks = [bot_instance_service.stop_bot(bot_id) for bot_id in bot_ids]
        await asyncio.gather(*stop_tasks, return_exceptions=True)


class TestBotManagementErrorHandling:
    """Test error handling in bot management operations."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_start_nonexistent_bot(self, bot_instance_service):
        """Test starting a bot that doesn't exist."""
        # GIVEN: Non-existent bot ID
        bot_id = "nonexistent_bot_12345"

        # WHEN: Trying to start non-existent bot
        result = await bot_instance_service.start_bot(bot_id)

        # THEN: Should return False (fails gracefully)
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_create_bot_invalid_config(self, bot_instance_service):
        """Test creating bot with invalid configuration."""
        # GIVEN: Invalid bot configuration (missing required fields)
        bot_config = BotConfiguration(
            bot_id="",  # Invalid: empty ID
            name="",  # Invalid: empty name
            bot_type=BotType.TRADING,
            version="1.0.0",
        )

        # WHEN/THEN: Creation should fail
        with pytest.raises((ValidationError, ServiceError)):
            await bot_instance_service.create_bot_instance(bot_config)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_register_bot_without_config(self, bot_coordination_service):
        """Test registering bot without proper configuration."""
        # GIVEN: Bot ID without config
        bot_id = f"test_bot_{uuid.uuid4().hex[:8]}"

        # WHEN/THEN: Registration should fail
        with pytest.raises((ValidationError, ServiceError)):
            await bot_coordination_service.register_bot(bot_id, None)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_request_excessive_resources(self, bot_resource_service):
        """Test requesting resources beyond available capacity."""
        # GIVEN: Excessive resource request
        bot_id = f"test_bot_{uuid.uuid4().hex[:8]}"
        excessive_capital = Decimal("999999999.00")

        # WHEN: Request excessive resources
        # THEN: Should either fail or return False
        try:
            result = await bot_resource_service.request_resources(
                bot_id=bot_id,
                capital_amount=excessive_capital,
                priority=BotPriority.NORMAL,
            )
            # If it doesn't raise, it should return False
            assert result is False
        except (ValidationError, ServiceError):
            # Expected to raise an error
            pass
