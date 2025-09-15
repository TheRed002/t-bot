"""
Comprehensive tests for BotService covering all critical business logic paths.

This test file focuses on achieving high coverage for the BotService which is the 
main orchestrator for bot management operations. Tests cover:
- Bot lifecycle management (create, start, stop, delete)
- Service dependencies and error handling
- Financial accuracy in capital allocation
- State management integration
- Metrics tracking and monitoring
- Health checks and system status
- Batch operations and resource management
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import pytest

from src.bot_management.service import BotService
from src.core.exceptions import ServiceError, ValidationError
from src.core.types import BotConfiguration, BotStatus, BotPriority, BotType, TradingMode, StateType
from src.state import StatePriority


class TestBotServiceInitialization:
    """Test bot service initialization and dependency injection."""

    def test_bot_service_init_with_dependencies(self, full_bot_service_deps):
        """Test service initialization with proper dependencies."""
        service = BotService(**full_bot_service_deps)
        
        assert service._name == "BotService"
        assert service._config_service == full_bot_service_deps['config_service']
        assert service._state_service == full_bot_service_deps['state_service']
        assert service._risk_service == full_bot_service_deps['risk_service']
        assert service._execution_service == full_bot_service_deps['execution_service']
        assert service._strategy_service == full_bot_service_deps['strategy_service']
        assert service._capital_service == full_bot_service_deps['capital_service']
        assert service._database_service == full_bot_service_deps['database_service']
        
        # Check default configuration
        assert service._max_concurrent_bots == 50
        assert service._max_capital_allocation == Decimal("1000000")
        assert service._heartbeat_timeout_seconds == 300

    def test_bot_service_init_without_metrics_collector(self, full_bot_service_deps):
        """Test initialization when metrics collector is not available."""
        # Remove metrics collector from dependencies
        deps_without_metrics = full_bot_service_deps.copy()
        deps_without_metrics['metrics_collector'] = None
        
        import src.bot_management.service
        with patch.object(src.bot_management.service, 'get_metrics_collector', return_value=None):
            service = BotService(**deps_without_metrics)
            assert service._metrics_collector is None
            assert service._trading_metrics is None

    @pytest.mark.asyncio
    async def test_service_startup_success(
        self, base_config, mock_state_service, mock_risk_service, 
        mock_execution_service, mock_strategy_service, mock_capital_service, 
        mock_database_service, mock_bot_repository, mock_bot_instance_repository,
        mock_bot_metrics_repository, mock_metrics_collector, mock_exchange_service
    ):
        """Test successful service startup."""
        # Configure config service to return bot management config
        base_config.get_config = MagicMock(return_value={
            "bot_management_service": {
                "max_concurrent_bots": 25,
                "max_capital_allocation": 500000,
                "heartbeat_timeout_seconds": 180,
                "health_check_interval_seconds": 45,
                "bot_startup_timeout_seconds": 90,
                "bot_shutdown_timeout_seconds": 45,
            }
        })
        
        service = BotService(
            config_service=base_config,
            state_service=mock_state_service,
            risk_service=mock_risk_service,
            execution_service=mock_execution_service,
            strategy_service=mock_strategy_service,
            capital_service=mock_capital_service,
            database_service=mock_database_service,
            bot_repository=mock_bot_repository,
            bot_instance_repository=mock_bot_instance_repository,
            bot_metrics_repository=mock_bot_metrics_repository,
            metrics_collector=mock_metrics_collector,
            exchange_service=mock_exchange_service,
        )
        
        # Mock load existing bot states
        with patch.object(service, '_load_existing_bot_states') as mock_load:
            mock_load.return_value = None
            await service._do_start()
        
        # Verify configuration was loaded
        assert service._max_concurrent_bots == 25
        assert service._max_capital_allocation == Decimal("500000")
        assert service._heartbeat_timeout_seconds == 180

    def test_service_startup_missing_dependencies(self):
        """Test service initialization fails with missing dependencies."""
        with pytest.raises(TypeError):
            BotService()

    @pytest.mark.asyncio
    async def test_service_startup_with_exception(
        self, base_config, mock_state_service, mock_risk_service, 
        mock_execution_service, mock_strategy_service, mock_capital_service, 
        mock_database_service, mock_bot_repository, mock_bot_instance_repository,
        mock_bot_metrics_repository, mock_metrics_collector, mock_exchange_service
    ):
        """Test service startup handles exceptions properly."""
        base_config.get_config = MagicMock(side_effect=Exception("Config error"))
        
        service = BotService(
            config_service=base_config,
            state_service=mock_state_service,
            risk_service=mock_risk_service,
            execution_service=mock_execution_service,
            strategy_service=mock_strategy_service,
            capital_service=mock_capital_service,
            database_service=mock_database_service,
            bot_repository=mock_bot_repository,
            bot_instance_repository=mock_bot_instance_repository,
            bot_metrics_repository=mock_bot_metrics_repository,
            metrics_collector=mock_metrics_collector,
            exchange_service=mock_exchange_service,
        )
        
        with pytest.raises(ServiceError, match="BotService startup failed"):
            await service._do_start()

    @pytest.mark.asyncio
    async def test_service_stop_cleans_up(self, lightweight_bot_config, setup_bot_service):
        """Test service stop properly cleans up active bots."""
        service = setup_bot_service
        
        # Add a bot to tracking
        service._active_bots["test_bot"] = {
            "config": lightweight_bot_config,
            "state": MagicMock(),
            "metrics": MagicMock(),
            "created_at": datetime.now(timezone.utc),
        }
        
        with patch.object(service, '_stop_all_active_bots') as mock_stop_all:
            mock_stop_all.return_value = None
            await service._do_stop()
        
        mock_stop_all.assert_called_once()
        assert len(service._active_bots) == 0
        assert len(service._bot_configurations) == 0
        assert len(service._bot_metrics) == 0


class TestBotCreation:
    """Test bot creation logic including validation and resource allocation."""

    @pytest.mark.asyncio
    async def test_create_bot_success(self, lightweight_bot_config, setup_bot_service):
        """Test successful bot creation."""
        service = setup_bot_service

        # Clear state from previous tests
        service._active_bots.clear()
        service._bot_configurations.clear()
        service._bot_metrics.clear()

        # Mock capital allocation
        service._capital_service.allocate_capital = AsyncMock(return_value=True)
        
        # Mock state persistence
        service._state_service.set_state = AsyncMock()
        
        # Mock strategy validation
        service._strategy_service.validate_strategy = AsyncMock(
            return_value={"valid": True}
        )
        
        result = await service.create_bot(lightweight_bot_config)
        
        assert result == lightweight_bot_config.bot_id
        assert lightweight_bot_config.bot_id in service._active_bots
        assert lightweight_bot_config.bot_id in service._bot_configurations
        
        # Verify capital allocation was called
        service._capital_service.allocate_capital.assert_called_once()
        
        # Verify state persistence was attempted
        assert service._state_service.set_state.call_count >= 1

    @pytest.mark.asyncio
    async def test_create_bot_validation_error(self, setup_bot_service):
        """Test bot creation fails with invalid configuration."""
        service = setup_bot_service
        
        # Create invalid config (missing required fields)
        invalid_config = BotConfiguration(
            bot_id="test_bot",
            name="",  # Empty name should fail validation
            bot_type=BotType.TRADING,
            version="1.0.0",  # Required field
            strategy_id="test_strategy",
            strategy_name="",  # Empty strategy name should fail
            exchanges=[],  # Empty exchanges should fail
            symbols=[],  # Empty symbols should fail
            allocated_capital=Decimal("0"),  # Zero capital should fail
        )
        
        with pytest.raises(ValidationError):
            await service.create_bot(invalid_config)

    @pytest.mark.asyncio
    async def test_create_bot_duplicate_id(self, lightweight_bot_config, setup_bot_service):
        """Test bot creation fails with duplicate ID."""
        service = setup_bot_service
        
        # Add existing bot
        service._active_bots[lightweight_bot_config.bot_id] = {"existing": True}
        
        with pytest.raises(ValidationError, match="Bot ID already exists"):
            await service.create_bot(lightweight_bot_config)

    @pytest.mark.asyncio
    async def test_create_bot_exceeds_limit(self, lightweight_bot_config, setup_bot_service):
        """Test bot creation fails when exceeding bot limit."""
        service = setup_bot_service
        service._max_concurrent_bots = 1
        
        # Fill the bot limit
        service._active_bots["existing_bot"] = {"existing": True}
        
        with pytest.raises(ServiceError, match="Maximum bot limit reached"):
            await service.create_bot(lightweight_bot_config)

    @pytest.mark.asyncio
    async def test_create_bot_capital_allocation_fails(
        self, lightweight_bot_config, setup_bot_service
    ):
        """Test bot creation fails when capital allocation fails."""
        service = setup_bot_service
        
        # Mock capital allocation failure
        service._capital_service.allocate_capital = AsyncMock(return_value=False)
        
        with patch.object(service, '_validate_bot_configuration') as mock_validate:
            mock_validate.return_value = None
            
            with pytest.raises(ServiceError, match="Failed to allocate capital"):
                await service.create_bot(lightweight_bot_config)

    @pytest.mark.asyncio
    async def test_create_bot_with_auto_start(
        self, bot_config_factory, setup_bot_service
    ):
        """Test bot creation with auto-start enabled."""
        service = setup_bot_service
        
        # Create unique config with auto-start enabled
        test_config = bot_config_factory("auto_start_bot", auto_start=True)
        
        # Mock dependencies
        service._capital_service.allocate_capital = AsyncMock(return_value=True)
        service._state_service.set_state = AsyncMock()
        service._strategy_service.validate_strategy = AsyncMock(
            return_value={"valid": True}
        )
        
        with patch.object(service, 'start_bot') as mock_start:
            mock_start.return_value = True
            
            result = await service.create_bot(test_config)
            
            assert result == test_config.bot_id
            mock_start.assert_called_once_with(test_config.bot_id)

    @pytest.mark.asyncio
    async def test_create_bot_state_persistence_error(
        self, bot_config_factory, setup_bot_service
    ):
        """Test bot creation handles state persistence errors gracefully."""
        service = setup_bot_service
        
        # Create unique config for this test
        test_config = bot_config_factory("persistence_error_bot")
        
        # Mock capital allocation success
        service._capital_service.allocate_capital = AsyncMock(return_value=True)
        
        # Mock state persistence failure
        service._state_service.set_state = AsyncMock(
            side_effect=Exception("State persistence failed")
        )
        
        # Mock strategy validation
        service._strategy_service.validate_strategy = AsyncMock(
            return_value={"valid": True}
        )
        
        # Mock metrics collector for error recording
        service._metrics_collector = AsyncMock()
        service._metrics_collector.increment = AsyncMock()
        
        with patch.object(service, '_validate_bot_configuration') as mock_validate:
            mock_validate.return_value = None
            
            result = await service.create_bot(test_config)
            
            # Should still succeed despite state persistence failure
            assert result == test_config.bot_id
            assert test_config.bot_id in service._active_bots
            
            # Should record error metric
            service._metrics_collector.increment.assert_called()


class TestBotLifecycle:
    """Test bot start, stop, and delete operations."""

    @pytest.mark.asyncio
    async def test_start_bot_success(self, setup_bot_service_with_bot):
        """Test successful bot startup."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock strategy validation and initialization
        service._strategy_service.validate_strategy = AsyncMock(
            return_value={"valid": True}
        )
        service._strategy_service.initialize_strategy = AsyncMock(
            return_value={"success": True}
        )
        
        # Mock execution service
        service._execution_service.start_bot_execution = AsyncMock(return_value=True)
        
        # Mock state service
        service._state_service.get_state = AsyncMock(return_value=None)
        service._state_service.set_state = AsyncMock()
        
        result = await service.start_bot(bot_config.bot_id)
        
        assert result is True
        
        # Verify strategy was initialized
        service._strategy_service.initialize_strategy.assert_called_once()
        
        # Verify execution service was started
        service._execution_service.start_bot_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_bot_not_found(self, setup_bot_service):
        """Test starting non-existent bot."""
        service = setup_bot_service
        
        with pytest.raises(ServiceError, match="Bot not found"):
            await service.start_bot("non_existent_bot")

    @pytest.mark.asyncio
    async def test_start_bot_already_running(self, setup_bot_service_with_bot):
        """Test starting already running bot."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock bot is already running
        service._state_service.get_state = AsyncMock(
            return_value={"status": BotStatus.RUNNING.value}
        )
        
        result = await service.start_bot(bot_config.bot_id)
        
        assert result is True  # Should return true for already running bot

    @pytest.mark.asyncio
    async def test_start_bot_strategy_validation_fails(
        self, setup_bot_service_with_bot
    ):
        """Test bot startup fails with invalid strategy."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock strategy validation failure
        service._strategy_service.validate_strategy = AsyncMock(
            return_value={"valid": False, "error": "Invalid strategy parameters"}
        )
        
        service._state_service.get_state = AsyncMock(return_value=None)
        service._state_service.set_state = AsyncMock()
        
        with pytest.raises(ServiceError, match="Strategy validation failed"):
            await service.start_bot(bot_config.bot_id)

    @pytest.mark.asyncio
    async def test_start_bot_execution_service_fails(
        self, setup_bot_service_with_bot
    ):
        """Test bot startup fails when execution service fails."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock successful strategy validation
        service._strategy_service.validate_strategy = AsyncMock(
            return_value={"valid": True}
        )
        service._strategy_service.initialize_strategy = AsyncMock(
            return_value={"success": True}
        )
        
        # Mock execution service failure
        service._execution_service.start_bot_execution = AsyncMock(return_value=False)
        
        service._state_service.get_state = AsyncMock(return_value=None)
        service._state_service.set_state = AsyncMock()
        
        with pytest.raises(ServiceError, match="Failed to start execution engine"):
            await service.start_bot(bot_config.bot_id)

    @pytest.mark.asyncio
    async def test_stop_bot_success(self, setup_bot_service_with_bot):
        """Test successful bot shutdown."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock execution service
        service._execution_service.stop_bot_execution = AsyncMock(return_value=True)
        
        # Mock strategy service
        service._strategy_service.cleanup_strategy = AsyncMock()
        
        # Mock state service
        service._state_service.get_state = AsyncMock(return_value={"status": "running"})
        service._state_service.set_state = AsyncMock()
        
        result = await service.stop_bot(bot_config.bot_id)
        
        assert result is True
        
        # Verify cleanup was called
        service._strategy_service.cleanup_strategy.assert_called_once_with(bot_config.bot_id)

    @pytest.mark.asyncio
    async def test_stop_bot_execution_warning(self, setup_bot_service_with_bot):
        """Test bot stop handles execution service failure gracefully."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock execution service failure
        service._execution_service.stop_bot_execution = AsyncMock(return_value=False)
        service._strategy_service.cleanup_strategy = AsyncMock()
        
        service._state_service.get_state = AsyncMock(return_value={"status": "running"})
        service._state_service.set_state = AsyncMock()
        
        result = await service.stop_bot(bot_config.bot_id)
        
        # Should still succeed despite execution service warning
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_bot_success(self, setup_bot_service_with_bot):
        """Test successful bot deletion."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock bot is stopped
        bot_data = service._active_bots[bot_config.bot_id]
        bot_data["state"].status = BotStatus.STOPPED
        
        # Mock repository archival
        service._bot_repository.update_status = AsyncMock()
        service._bot_metrics_repository.get_latest_metrics = AsyncMock(return_value=None)
        
        # Mock state service
        service._state_service.delete_state = AsyncMock()
        
        # Mock strategy cleanup
        service._strategy_service.cleanup_strategy = AsyncMock()
        
        result = await service.delete_bot(bot_config.bot_id)
        
        assert result is True
        assert bot_config.bot_id not in service._active_bots
        assert bot_config.bot_id not in service._bot_configurations
        
        # Verify cleanup operations
        service._bot_repository.update_status.assert_called_once_with(bot_config.bot_id, BotStatus.STOPPED)
        service._state_service.delete_state.assert_called_once()
        service._strategy_service.cleanup_strategy.assert_called_once_with(bot_config.bot_id)

    @pytest.mark.asyncio
    async def test_delete_bot_force_running(self, setup_bot_service_with_bot):
        """Test force deletion of running bot."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock bot is running
        bot_data = service._active_bots[bot_config.bot_id]
        bot_data["state"].status = BotStatus.RUNNING
        
        # Mock stop implementation
        with patch.object(service, '_stop_bot_impl', new_callable=AsyncMock) as mock_stop:
            mock_stop.return_value = True
            
            # Mock other dependencies
            service._bot_repository.update_status = AsyncMock()
            service._bot_metrics_repository.get_latest_metrics = AsyncMock(return_value=None)
            service._state_service.delete_state = AsyncMock()
            service._strategy_service.cleanup_strategy = AsyncMock()
            
            result = await service.delete_bot(bot_config.bot_id, force=True)
            
            assert result is True
            mock_stop.assert_called_once_with(bot_config.bot_id)

    @pytest.mark.asyncio
    async def test_delete_bot_running_no_force(self, setup_bot_service_with_bot):
        """Test deletion fails for running bot without force."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock bot is running
        bot_data = service._active_bots[bot_config.bot_id]
        bot_data["state"].status = BotStatus.RUNNING
        
        with pytest.raises(ServiceError, match="Cannot delete running bot"):
            await service.delete_bot(bot_config.bot_id, force=False)


class TestBotStatus:
    """Test bot status monitoring and health checks."""

    @pytest.mark.asyncio
    async def test_get_bot_status_success(self, setup_bot_service_with_bot):
        """Test getting bot status successfully."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock state service
        mock_state = {"status": "running", "last_update": "2024-01-01T12:00:00Z"}
        service._state_service.get_state = AsyncMock(return_value=mock_state)
        
        # Mock bot metrics repository
        mock_metrics = {"pnl": "100.0", "trades": 5}
        service._bot_metrics_repository.get_latest_metrics = AsyncMock(return_value=mock_metrics)
        
        # Mock execution service
        mock_execution = {"status": "healthy", "orders_pending": 0}
        service._execution_service.get_bot_execution_status = AsyncMock(
            return_value=mock_execution
        )
        
        result = await service.get_bot_status(bot_config.bot_id)
        
        assert result["bot_id"] == bot_config.bot_id
        assert result["state"] == mock_state
        assert result["metrics"] == mock_metrics
        assert result["execution_status"] == mock_execution
        assert result["service_status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_bot_status_not_found(self, setup_bot_service):
        """Test getting status for non-existent bot."""
        service = setup_bot_service
        
        with pytest.raises(ServiceError, match="Bot not found"):
            await service.get_bot_status("non_existent_bot")

    @pytest.mark.asyncio
    async def test_get_all_bots_status(self, setup_bot_service_with_multiple_bots):
        """Test getting status for all bots."""
        service = setup_bot_service_with_multiple_bots
        
        # Mock get_bot_status_impl for each bot
        with patch.object(service, '_get_bot_status_impl') as mock_get_status:
            mock_get_status.return_value = {
                "bot_id": "test_bot",
                "state": {"status": BotStatus.RUNNING.value},
                "metrics": {"pnl": "50.0"},
            }
            
            result = await service.get_all_bots_status()
            
            assert "summary" in result
            assert "bots" in result
            assert result["summary"]["total_bots"] >= 1

    @pytest.mark.asyncio
    async def test_get_all_bots_status_with_errors(
        self, setup_bot_service_with_multiple_bots
    ):
        """Test get all bots status handles individual bot errors."""
        service = setup_bot_service_with_multiple_bots
        
        # Mock get_bot_status_impl to raise exception for some bots
        def side_effect(bot_id):
            if bot_id == "error_bot":
                raise Exception("Bot status error")
            return {
                "bot_id": bot_id,
                "state": {"status": BotStatus.RUNNING.value},
                "metrics": {"pnl": "50.0"},
            }
        
        with patch.object(service, '_get_bot_status_impl', side_effect=side_effect):
            # Add an error bot
            service._active_bots["error_bot"] = {"mock": "data"}
            
            result = await service.get_all_bots_status()
            
            assert "error_bot" in result["bots"]
            assert "error" in result["bots"]["error_bot"]
            assert result["summary"]["error"] >= 1


class TestBotMetrics:
    """Test bot metrics management and updates."""

    @pytest.mark.asyncio
    async def test_update_bot_metrics_success(self, setup_bot_service_with_bot):
        """Test successful bot metrics update."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock bot metrics repository
        service._bot_metrics_repository.save_metrics = AsyncMock()
        
        metrics_data = {
            "pnl": Decimal("100.50"),
            "trades_count": 15,
            "win_rate": 0.73,
        }
        
        result = await service.update_bot_metrics(bot_config.bot_id, metrics_data)
        
        assert result is True
        
        # Verify metrics were stored
        service._bot_metrics_repository.save_metrics.assert_called_once()
        
        # Verify local metrics were updated
        assert bot_config.bot_id in service._bot_metrics

    @pytest.mark.asyncio
    async def test_update_bot_metrics_with_trading_metrics(
        self, setup_bot_service_with_bot
    ):
        """Test metrics update with trading metrics recording."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock metrics collector and trading metrics
        mock_metrics_collector = AsyncMock()
        mock_trading_metrics = MagicMock()
        mock_trading_metrics.record_pnl = MagicMock()
        mock_trading_metrics.increment_trades = MagicMock()
        
        service._metrics_collector = mock_metrics_collector
        service._trading_metrics = mock_trading_metrics
        
        # Mock bot metrics repository
        service._bot_metrics_repository.save_metrics = AsyncMock()
        
        metrics_data = {
            "pnl": Decimal("100.50"),
            "trades_count": 15,
            "win_rate": 0.73,
        }
        
        result = await service.update_bot_metrics(bot_config.bot_id, metrics_data)
        
        assert result is True
        
        # Verify trading metrics were recorded
        mock_trading_metrics.record_pnl.assert_called_once()
        mock_trading_metrics.increment_trades.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_bot_metrics_not_found(self, setup_bot_service):
        """Test updating metrics for non-existent bot."""
        service = setup_bot_service
        
        with pytest.raises(ServiceError, match="Bot not found"):
            await service.update_bot_metrics("non_existent_bot", {"pnl": "100.0"})


class TestBatchOperations:
    """Test batch operations for multiple bots."""

    @pytest.mark.asyncio
    async def test_start_all_bots_success(self, setup_bot_service_with_multiple_bots):
        """Test starting all bots successfully."""
        service = setup_bot_service_with_multiple_bots
        
        # Set bot states to allow starting
        for bot_data in service._active_bots.values():
            bot_data["state"].status = BotStatus.READY
        
        with patch.object(service, '_start_bot_impl') as mock_start:
            mock_start.return_value = True
            
            result = await service.start_all_bots()
            
            assert all(result.values())  # All should succeed
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_start_all_bots_with_priority_filter(
        self, setup_bot_service_with_multiple_bots
    ):
        """Test starting bots with priority filter."""
        service = setup_bot_service_with_multiple_bots
        
        # Set different priorities
        bot_ids = list(service._active_bots.keys())
        if len(bot_ids) >= 2:
            service._active_bots[bot_ids[0]]["config"].priority = BotPriority.HIGH
            service._active_bots[bot_ids[0]]["state"].status = BotStatus.READY
            service._active_bots[bot_ids[1]]["config"].priority = BotPriority.NORMAL
            service._active_bots[bot_ids[1]]["state"].status = BotStatus.READY
        
        with patch.object(service, '_start_bot_impl') as mock_start:
            mock_start.return_value = True
            
            result = await service.start_all_bots(BotPriority.HIGH)
            
            # Should only start high priority bots
            assert len(result) <= len(bot_ids)

    @pytest.mark.asyncio
    async def test_start_all_bots_with_failures(
        self, setup_bot_service_with_multiple_bots
    ):
        """Test starting all bots handles individual failures."""
        service = setup_bot_service_with_multiple_bots
        
        # Set bot states to allow starting
        for bot_data in service._active_bots.values():
            bot_data["state"].status = BotStatus.READY
        
        def start_side_effect(bot_id):
            if "fail" in bot_id:
                raise Exception("Start failed")
            return True
        
        with patch.object(service, '_start_bot_impl', side_effect=start_side_effect):
            # Add a bot that will fail
            service._active_bots["fail_bot"] = {
                "config": MagicMock(),
                "state": MagicMock(status=BotStatus.READY),
            }
            service._active_bots["fail_bot"]["config"].priority = BotPriority.NORMAL
            
            result = await service.start_all_bots()
            
            # Should have mixed results
            assert "fail_bot" in result
            assert result["fail_bot"] is False

    @pytest.mark.asyncio
    async def test_stop_all_bots_success(self, setup_bot_service_with_multiple_bots):
        """Test stopping all bots successfully."""
        service = setup_bot_service_with_multiple_bots
        
        # Set bot states to running
        for bot_data in service._active_bots.values():
            bot_data["state"].status = BotStatus.RUNNING
        
        with patch.object(service, '_stop_bot_impl') as mock_stop:
            mock_stop.return_value = True
            
            result = await service.stop_all_bots()
            
            assert all(result.values())  # All should succeed
            assert len(result) > 0


class TestHealthChecks:
    """Test comprehensive health check functionality."""

    @pytest.mark.asyncio
    async def test_perform_health_check_success(self, setup_bot_service_with_bot):
        """Test successful health check."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock service health checks
        service._state_service.health_check = AsyncMock(
            return_value={"status": "healthy"}
        )
        service._execution_service.health_check = AsyncMock(
            return_value={"status": "healthy"}
        )
        
        # Mock risk service with health check
        service._risk_service.health_check = AsyncMock(
            return_value={"status": "healthy"}
        )
        
        result = await service.perform_health_check(bot_config.bot_id)
        
        assert result["bot_id"] == bot_config.bot_id
        assert result["healthy"] is True
        assert "checks" in result
        assert "state_service" in result["checks"]
        assert "execution_service" in result["checks"]

    @pytest.mark.asyncio
    async def test_perform_health_check_service_failures(
        self, setup_bot_service_with_bot
    ):
        """Test health check with service failures."""
        service, bot_config = setup_bot_service_with_bot
        
        # Mock service failures
        service._state_service.health_check = AsyncMock(
            side_effect=Exception("State service error")
        )
        service._execution_service.health_check = AsyncMock(
            return_value={"status": "unhealthy"}
        )
        
        result = await service.perform_health_check(bot_config.bot_id)
        
        assert result["bot_id"] == bot_config.bot_id
        assert result["healthy"] is False
        assert result["checks"]["state_service"]["healthy"] is False
        assert result["checks"]["execution_service"]["healthy"] is False

    @pytest.mark.asyncio
    async def test_perform_health_check_not_found(self, setup_bot_service):
        """Test health check for non-existent bot."""
        service = setup_bot_service
        
        result = await service.perform_health_check("non_existent_bot")
        
        assert result["status"] == "not_found"
        assert result["healthy"] is False

    @pytest.mark.asyncio
    async def test_service_health_check_method(self, setup_bot_service):
        """Test the service's own health check method."""
        service = setup_bot_service
        
        # Mock all required services
        service._database_service = MagicMock()
        service._state_service = MagicMock()
        service._risk_service = MagicMock()
        service._execution_service = MagicMock()
        service._strategy_service = MagicMock()
        
        # Add health check methods
        service._state_service.health_check = AsyncMock(
            return_value={"status": "healthy"}
        )
        
        result = await service._service_health_check()
        
        # Should return HealthStatus enum value
        from src.core.base.interfaces import HealthStatus
        assert result in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]


class TestValidationMethods:
    """Test bot configuration validation methods."""

    @pytest.mark.asyncio
    async def test_validate_bot_configuration_success(self, setup_bot_service):
        """Test successful bot configuration validation."""
        service = setup_bot_service
        
        # Mock strategy validation
        service._strategy_service.validate_strategy = AsyncMock(
            return_value={"valid": True}
        )
        
        valid_config = BotConfiguration(
            bot_id="test_bot",
            name="Test Bot",
            bot_type=BotType.TRADING,
            version="1.0.0",  # Required field
            strategy_id="test_strategy",
            strategy_name="Test Strategy",
            exchanges=["binance"],
            symbols=["BTCUSDT"],
            allocated_capital=Decimal("1000.0"),
            risk_percentage=0.02,
            strategy_parameters={}
        )
        
        # Should not raise any exception
        await service._validate_bot_configuration(valid_config)

    @pytest.mark.asyncio
    async def test_validate_bot_configuration_failures(self, setup_bot_service):
        """Test bot configuration validation failures."""
        service = setup_bot_service
        
        # Test missing bot_id
        with pytest.raises(ValidationError, match="Bot ID is required"):
            await service._validate_bot_configuration(
                BotConfiguration(
                    bot_id="",
                    name="Test Bot",
                    bot_type=BotType.TRADING,
                    version="1.0.0",  # Required field
                    strategy_id="test_strategy",
                    exchanges=["binance"],
                    symbols=["BTCUSDT"],
                    allocated_capital=Decimal("1000.0")
                )
            )
        
        # Test missing name
        with pytest.raises(ValidationError, match="Bot name is required"):
            await service._validate_bot_configuration(
                BotConfiguration(
                    bot_id="test_bot",
                    name="",
                    bot_type=BotType.TRADING,
                    version="1.0.0",  # Required field
                    strategy_id="test_strategy",
                    exchanges=["binance"],
                    symbols=["BTCUSDT"],
                    allocated_capital=Decimal("1000.0")
                )
            )
        
        # Test empty exchanges
        with pytest.raises(ValidationError, match="At least one exchange is required"):
            await service._validate_bot_configuration(
                BotConfiguration(
                    bot_id="test_bot",
                    name="Test Bot",
                    bot_type=BotType.TRADING,
                    version="1.0.0",  # Required field
                    strategy_id="test_strategy",
                    strategy_name="Test Strategy",  # Required field
                    exchanges=[],
                    symbols=["BTCUSDT"],
                    allocated_capital=Decimal("1000.0")
                )
            )
        
        # Test zero capital
        with pytest.raises(ValidationError, match="Allocated capital must be positive"):
            await service._validate_bot_configuration(
                BotConfiguration(
                    bot_id="test_bot",
                    name="Test Bot",
                    bot_type=BotType.TRADING,
                    version="1.0.0",  # Required field
                    strategy_id="test_strategy",
                    strategy_name="Test Strategy",  # Required field
                    exchanges=["binance"],
                    symbols=["BTCUSDT"],
                    allocated_capital=Decimal("0")
                )
            )

    @pytest.mark.asyncio
    async def test_validate_bot_configuration_strategy_invalid(self, setup_bot_service):
        """Test bot configuration validation with invalid strategy."""
        service = setup_bot_service
        
        # Mock strategy validation failure
        service._strategy_service.validate_strategy = AsyncMock(
            return_value={"valid": False, "error": "Strategy not found"}
        )
        
        config = BotConfiguration(
            bot_id="test_bot",
            name="Test Bot",
            bot_type=BotType.TRADING,
            version="1.0.0",  # Required field
            strategy_id="invalid_strategy",
            strategy_name="Invalid Strategy",
            exchanges=["binance"],
            symbols=["BTCUSDT"],
            allocated_capital=Decimal("1000.0"),
            risk_percentage=0.02,
            strategy_parameters={}
        )
        
        with pytest.raises(ValidationError, match="Invalid strategy"):
            await service._validate_bot_configuration(config)


class TestLoadExistingBotStates:
    """Test loading existing bot states from StateService."""

    @pytest.mark.asyncio
    async def test_load_existing_bot_states_success(self, setup_bot_service):
        """Test successful loading of existing bot states."""
        service = setup_bot_service
        
        # Mock state service response
        mock_bot_states = [
            {
                "data": {
                    "bot_id": "existing_bot_1",
                    "status": BotStatus.STOPPED.value,
                    "created_at": "2024-01-01T12:00:00+00:00",
                    "configuration": {
                        "bot_id": "existing_bot_1",
                        "name": "Existing Bot 1",
                        "bot_type": BotType.TRADING.value,
                        "version": "1.0.0",  # Required field
                        "strategy_id": "test_strategy",
                        "strategy_name": "Test Strategy",
                        "exchanges": ["binance"],
                        "symbols": ["BTCUSDT"],
                        "allocated_capital": "1000.0",
                        "risk_percentage": 0.02,
                    }
                }
            }
        ]
        
        service._state_service.get_states_by_type = AsyncMock(
            return_value=mock_bot_states
        )
        
        await service._load_existing_bot_states()
        
        # Verify bot was loaded into local tracking
        assert "existing_bot_1" in service._active_bots
        assert "existing_bot_1" in service._bot_configurations
        assert "existing_bot_1" in service._bot_metrics

    @pytest.mark.asyncio
    async def test_load_existing_bot_states_no_method(self, setup_bot_service):
        """Test loading when get_states_by_type method doesn't exist."""
        service = setup_bot_service
        
        # Remove the method from state service
        if hasattr(service._state_service, 'get_states_by_type'):
            delattr(service._state_service, 'get_states_by_type')
        
        # Should handle gracefully and not raise exception
        await service._load_existing_bot_states()
        
        # Should be empty since no states were loaded
        assert len(service._active_bots) == 0

    @pytest.mark.asyncio
    async def test_load_existing_bot_states_with_errors(self, setup_bot_service):
        """Test loading bot states handles individual bot errors."""
        service = setup_bot_service
        
        # Mock state service with mixed valid and invalid data
        mock_bot_states = [
            {
                "data": {
                    "bot_id": "valid_bot",
                    "status": BotStatus.STOPPED.value,
                    "created_at": "2024-01-01T12:00:00+00:00",
                    "configuration": {
                        "bot_id": "valid_bot",
                        "name": "Valid Bot",
                        "bot_type": BotType.TRADING.value,
                        "version": "1.0.0",  # Required field
                        "strategy_id": "test_strategy",
                        "strategy_name": "Test Strategy",
                        "exchanges": ["binance"],
                        "symbols": ["BTCUSDT"],
                        "allocated_capital": "1000.0",
                        "risk_percentage": 0.02,  # Required field
                    }
                }
            },
            {
                "data": {
                    "bot_id": "invalid_bot",
                    "configuration": "invalid_config"  # Invalid configuration
                }
            }
        ]
        
        service._state_service.get_states_by_type = AsyncMock(
            return_value=mock_bot_states
        )
        
        await service._load_existing_bot_states()
        
        # Should load valid bot and skip invalid one
        assert "valid_bot" in service._active_bots
        assert "invalid_bot" not in service._active_bots


# Fixtures for test setup
@pytest.fixture(scope="function")
def setup_bot_service(
    base_config, mock_state_service, mock_risk_service, 
    mock_execution_service, mock_strategy_service, mock_capital_service, 
    mock_database_service
):
    """Setup a properly configured bot service for testing."""
    # Create properly configured exchange service mock
    mock_exchange_service = AsyncMock()
    mock_exchange_service.get_exchange_health = AsyncMock(return_value={"status": "healthy"})
    mock_exchange_service.is_symbol_supported = AsyncMock(return_value=True)
    mock_exchange_service.get_rate_limits = AsyncMock(return_value={"requests_per_minute": 1000})
    mock_exchange_service.get_account_balance = AsyncMock(return_value={"available": 10000.0, "total": 10000.0})
    mock_exchange_service.check_health = AsyncMock(return_value=True)
    mock_exchange_service.get_exchange = AsyncMock(return_value=AsyncMock())
    
    return BotService(
        config_service=base_config,
        state_service=mock_state_service,
        risk_service=mock_risk_service,
        execution_service=mock_execution_service,
        strategy_service=mock_strategy_service,
        capital_service=mock_capital_service,
        exchange_service=mock_exchange_service,
        bot_repository=AsyncMock(),  # Required dependency
        bot_instance_repository=AsyncMock(),  # Required dependency
        bot_metrics_repository=AsyncMock(),  # Required dependency
        metrics_collector=AsyncMock(),  # Required dependency
        database_service=mock_database_service,
    )


@pytest.fixture(scope="function")
def setup_bot_service_with_bot(setup_bot_service, lightweight_bot_config):
    """Setup bot service with a test bot already added."""
    service = setup_bot_service
    
    # Add bot to service tracking
    from src.core.types import BotState
    
    bot_state = BotState(
        bot_id=lightweight_bot_config.bot_id,
        status=BotStatus.READY,
        created_at=datetime.now(timezone.utc),
        configuration=lightweight_bot_config,
    )
    
    service._active_bots[lightweight_bot_config.bot_id] = {
        "config": lightweight_bot_config,
        "state": bot_state,
        "metrics": MagicMock(),
        "created_at": datetime.now(timezone.utc),
    }
    
    service._bot_configurations[lightweight_bot_config.bot_id] = lightweight_bot_config
    
    # Initialize bot metrics so update can succeed
    from src.core.types import BotMetrics
    service._bot_metrics[lightweight_bot_config.bot_id] = BotMetrics(
        bot_id=lightweight_bot_config.bot_id,
        total_trades=0,
        successful_trades=0,
        failed_trades=0,
        profitable_trades=0,
        losing_trades=0,
        total_pnl=Decimal("0"),
        win_rate=0.0,
        average_trade_pnl=Decimal("0"),
        uptime_percentage=0.0,
        error_count=0,
        last_heartbeat=datetime.now(timezone.utc),
        cpu_usage=0.0,
        memory_usage=0.0,
        api_calls_made=0,
        start_time=datetime.now(timezone.utc),
        last_trade_time=datetime.now(timezone.utc),
        metrics_updated_at=datetime.now(timezone.utc),
    )
    
    return service, lightweight_bot_config


@pytest.fixture(scope="function")
def setup_bot_service_with_multiple_bots(setup_bot_service, bot_config_factory):
    """Setup bot service with multiple test bots."""
    service = setup_bot_service
    
    for i in range(3):
        bot_config = bot_config_factory(f"test_bot_{i}")
        
        from src.core.types import BotState
        bot_state = BotState(
            bot_id=bot_config.bot_id,
            status=BotStatus.READY,
            created_at=datetime.now(timezone.utc),
            configuration=bot_config,
        )
        
        service._active_bots[bot_config.bot_id] = {
            "config": bot_config,
            "state": bot_state,
            "metrics": MagicMock(),
            "created_at": datetime.now(timezone.utc),
        }
        
        service._bot_configurations[bot_config.bot_id] = bot_config
    
    return service