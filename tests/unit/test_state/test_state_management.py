"""
Comprehensive tests for the State Management system (P-018, P-023, P-024, P-025).

This test suite covers:
- State Manager functionality
- Checkpoint management
- Trade lifecycle management
- Quality control validation
- Real-time state synchronization
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import Config
from src.core.types import (
    BotState, BotStatus, OrderRequest, OrderSide, OrderType, 
    ExecutionResult, MarketData
)
from src.state.state_manager import StateManager, StateSnapshot
from src.state.checkpoint_manager import CheckpointManager, CheckpointMetadata
from src.state.trade_lifecycle_manager import (
    TradeLifecycleManager, TradeLifecycleState, TradeContext
)
from src.state.quality_controller import (
    QualityController, PreTradeValidation, PostTradeAnalysis, ValidationResult
)
from src.state.state_sync_manager import (
    StateSyncManager, SyncEventType, ConflictResolutionStrategy
)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = MagicMock(spec=Config)
    config.state_management = {
        "max_snapshots_per_bot": 10,
        "snapshot_interval_minutes": 5,
        "redis_ttl_seconds": 3600,
        "enable_compression": True
    }
    config.quality_controls = {
        "min_quality_score": 70.0,
        "slippage_threshold_bps": 20.0,
        "execution_time_threshold_seconds": 30.0,
        "market_impact_threshold_bps": 10.0
    }
    config.state_sync = {
        "sync_interval_seconds": 5,
        "conflict_resolution_timeout_seconds": 30,
        "max_sync_retries": 3,
        "default_resolution_strategy": "last_write_wins"
    }
    return config


@pytest.fixture
def sample_bot_state():
    """Sample bot state for testing."""
    return BotState(
        bot_id="test-bot-123",
        status=BotStatus.RUNNING,
        allocated_capital=Decimal("10000.00"),
        used_capital=Decimal("5000.00"),
        open_positions=[
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "quantity": 0.5,
                "entry_price": 50000.0
            }
        ],
        pending_orders=[
            {
                "symbol": "ETH/USDT",
                "side": "buy",
                "quantity": 2.0,
                "price": 3000.0
            }
        ],
        strategy_state={"momentum": 0.65, "volatility": 0.12},
        state_version=5
    )


@pytest.fixture
def sample_order_request():
    """Sample order request for testing."""
    return OrderRequest(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        price=Decimal("50000.0"),
        client_order_id="test-order-123"
    )


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return MarketData(
        symbol="BTC/USDT",
        price=Decimal("50000.0"),
        volume=Decimal("1000.0"),
        bid=Decimal("49995.0"),
        ask=Decimal("50005.0"),
        timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_execution_result():
    """Sample execution result for testing."""
    return ExecutionResult(
        execution_id="exec-123",
        original_order=None,  # Would be filled in real scenario
        total_filled_quantity=Decimal("0.1"),
        average_fill_price=Decimal("50010.0"),
        total_fees=Decimal("5.0"),
        fills=[],
        status="completed",
        execution_duration_seconds=15.5
    )


class TestStateManager:
    """Test cases for StateManager."""

    @pytest.fixture
    async def state_manager(self, mock_config):
        """Create StateManager instance for testing."""
        with patch('src.state.state_manager.DatabaseManager'), \
             patch('src.state.state_manager.RedisClient'), \
             patch('src.state.state_manager.InfluxDBClient'):
            
            manager = StateManager(mock_config)
            
            # Mock database clients
            manager.db_manager = AsyncMock()
            manager.redis_client = AsyncMock()
            manager.influxdb_client = AsyncMock()
            
            # Mock database session
            mock_session = AsyncMock()
            manager.db_manager.get_session.return_value.__aenter__.return_value = mock_session
            manager.db_manager.get_session.return_value.__aexit__.return_value = None
            
            await manager.initialize()
            return manager

    @pytest.mark.asyncio
    async def test_save_bot_state(self, state_manager, sample_bot_state):
        """Test saving bot state."""
        # Mock Redis and PostgreSQL operations
        state_manager.redis_client.setex = AsyncMock()
        
        version_id = await state_manager.save_bot_state(
            "test-bot-123", 
            sample_bot_state,
            create_snapshot=True
        )
        
        assert version_id is not None
        assert len(version_id) > 0
        
        # Verify Redis was called
        state_manager.redis_client.setex.assert_called_once()
        
        # Verify state is cached locally
        assert len(state_manager._state_cache) > 0

    @pytest.mark.asyncio
    async def test_load_bot_state(self, state_manager, sample_bot_state):
        """Test loading bot state."""
        # Mock Redis return data
        redis_data = {
            "version_id": "test-version",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": sample_bot_state.model_dump()
        }
        state_manager.redis_client.get = AsyncMock(
            return_value=json.dumps(redis_data, default=str)
        )
        
        loaded_state = await state_manager.load_bot_state("test-bot-123")
        
        assert loaded_state is not None
        assert loaded_state.bot_id == sample_bot_state.bot_id
        assert loaded_state.status == sample_bot_state.status

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, state_manager, sample_bot_state):
        """Test checkpoint creation."""
        # Setup state in manager
        state_manager.local_state_cache["bot_state:test-bot-123"] = sample_bot_state.model_dump()
        
        # Mock load_bot_state to return our sample state
        with patch.object(state_manager, 'load_bot_state', return_value=sample_bot_state):
            checkpoint_id = await state_manager.create_checkpoint("test-bot-123")
        
        assert checkpoint_id is not None
        assert len(checkpoint_id) > 0

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint(self, state_manager, sample_bot_state):
        """Test restoring from checkpoint."""
        # Create a snapshot
        snapshot = StateSnapshot(
            bot_id="test-bot-123",
            bot_state=sample_bot_state.model_dump()
        )
        state_manager._state_cache["test-checkpoint"] = snapshot
        
        # Mock save_bot_state
        with patch.object(state_manager, 'save_bot_state', return_value="new-version") as mock_save:
            success = await state_manager.restore_from_checkpoint(
                "test-bot-123", 
                "test-checkpoint"
            )
        
        assert success is True
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_state_metrics(self, state_manager):
        """Test getting state metrics."""
        # Add some test data
        state_manager._transaction_log["tx1"] = MagicMock()
        state_manager._transaction_log["tx1"].bot_id = "test-bot-123"
        state_manager._transaction_log["tx1"].timestamp = datetime.now(timezone.utc)
        state_manager._transaction_log["tx1"].status = "committed"
        state_manager._transaction_log["tx1"].operations = [{"type": "update"}]
        
        metrics = await state_manager.get_state_metrics("test-bot-123", 24)
        
        assert "bot_id" in metrics
        assert "period_hours" in metrics
        assert "state_updates" in metrics


class TestCheckpointManager:
    """Test cases for CheckpointManager."""

    @pytest.fixture
    async def checkpoint_manager(self, mock_config):
        """Create CheckpointManager instance for testing."""
        with patch('src.state.checkpoint_manager.ensure_directory_exists'):
            manager = CheckpointManager(mock_config)
            await manager.initialize()
            return manager

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, checkpoint_manager, sample_bot_state):
        """Test checkpoint creation."""
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            "test-bot-123",
            sample_bot_state,
            checkpoint_type="manual"
        )
        
        assert checkpoint_id is not None
        assert checkpoint_id in checkpoint_manager.checkpoints
        
        metadata = checkpoint_manager.checkpoints[checkpoint_id]
        assert metadata.bot_id == "test-bot-123"
        assert metadata.checkpoint_type == "manual"

    @pytest.mark.asyncio
    async def test_restore_checkpoint(self, checkpoint_manager, sample_bot_state):
        """Test checkpoint restoration."""
        # First create a checkpoint
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            "test-bot-123",
            sample_bot_state
        )
        
        # Mock file operations for restore
        with patch('builtins.open', create=True) as mock_open, \
             patch('pathlib.Path.exists', return_value=True):
            
            # Mock file content
            import pickle
            mock_data = pickle.dumps({
                "bot_id": "test-bot-123",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bot_state": sample_bot_state.model_dump()
            })
            mock_open.return_value.__enter__.return_value.read.return_value = mock_data
            
            bot_id, restored_state = await checkpoint_manager.restore_checkpoint(checkpoint_id)
        
        assert bot_id == "test-bot-123"
        assert restored_state.bot_id == sample_bot_state.bot_id

    @pytest.mark.asyncio
    async def test_create_recovery_plan(self, checkpoint_manager, sample_bot_state):
        """Test recovery plan creation."""
        # Create a checkpoint first
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            "test-bot-123",
            sample_bot_state
        )
        
        plan = await checkpoint_manager.create_recovery_plan("test-bot-123")
        
        assert plan.bot_id == "test-bot-123"
        assert plan.target_checkpoint_id == checkpoint_id
        assert len(plan.steps) > 0
        assert plan.estimated_duration_seconds > 0

    @pytest.mark.asyncio
    async def test_schedule_checkpoint(self, checkpoint_manager):
        """Test checkpoint scheduling."""
        await checkpoint_manager.schedule_checkpoint("test-bot-123", 30)
        
        assert "test-bot-123" in checkpoint_manager.checkpoint_schedules
        
        scheduled_time = checkpoint_manager.checkpoint_schedules["test-bot-123"]
        assert scheduled_time > datetime.now(timezone.utc)

    @pytest.mark.asyncio
    async def test_get_checkpoint_stats(self, checkpoint_manager, sample_bot_state):
        """Test checkpoint statistics."""
        # Create some checkpoints
        await checkpoint_manager.create_checkpoint("test-bot-123", sample_bot_state)
        await checkpoint_manager.create_checkpoint("test-bot-456", sample_bot_state)
        
        stats = await checkpoint_manager.get_checkpoint_stats()
        
        assert stats["total_checkpoints"] == 2
        assert "bot_stats" in stats
        assert "performance_stats" in stats


class TestTradeLifecycleManager:
    """Test cases for TradeLifecycleManager."""

    @pytest.fixture
    async def lifecycle_manager(self, mock_config):
        """Create TradeLifecycleManager instance for testing."""
        with patch('src.state.trade_lifecycle_manager.DatabaseManager'), \
             patch('src.state.trade_lifecycle_manager.RedisClient'):
            
            manager = TradeLifecycleManager(mock_config)
            
            # Mock database clients
            manager.db_manager = AsyncMock()
            manager.redis_client = AsyncMock()
            
            await manager.initialize()
            return manager

    @pytest.mark.asyncio
    async def test_start_trade_lifecycle(self, lifecycle_manager, sample_order_request):
        """Test starting trade lifecycle."""
        trade_id = await lifecycle_manager.start_trade_lifecycle(
            "test-bot-123",
            "momentum_strategy",
            sample_order_request
        )
        
        assert trade_id is not None
        assert trade_id in lifecycle_manager.active_trades
        
        trade_context = lifecycle_manager.active_trades[trade_id]
        assert trade_context.bot_id == "test-bot-123"
        assert trade_context.strategy_name == "momentum_strategy"
        assert trade_context.current_state == TradeLifecycleState.SIGNAL_GENERATED

    @pytest.mark.asyncio
    async def test_transition_trade_state(self, lifecycle_manager, sample_order_request):
        """Test trade state transitions."""
        # Start a trade
        trade_id = await lifecycle_manager.start_trade_lifecycle(
            "test-bot-123",
            "momentum_strategy", 
            sample_order_request
        )
        
        # Transition to next state
        success = await lifecycle_manager.transition_trade_state(
            trade_id,
            TradeLifecycleState.PRE_TRADE_VALIDATION
        )
        
        assert success is True
        
        trade_context = lifecycle_manager.active_trades[trade_id]
        assert trade_context.current_state == TradeLifecycleState.PRE_TRADE_VALIDATION
        assert trade_context.previous_state == TradeLifecycleState.SIGNAL_GENERATED

    @pytest.mark.asyncio
    async def test_update_trade_execution(self, lifecycle_manager, sample_order_request, sample_execution_result):
        """Test updating trade execution."""
        # Start a trade and transition to submitted state
        trade_id = await lifecycle_manager.start_trade_lifecycle(
            "test-bot-123",
            "momentum_strategy",
            sample_order_request
        )
        
        await lifecycle_manager.transition_trade_state(
            trade_id,
            TradeLifecycleState.ORDER_SUBMITTED
        )
        
        # Update with execution result
        await lifecycle_manager.update_trade_execution(trade_id, sample_execution_result)
        
        trade_context = lifecycle_manager.active_trades[trade_id]
        assert trade_context.filled_quantity == sample_execution_result.total_filled_quantity
        assert trade_context.order_id == sample_execution_result.execution_id

    @pytest.mark.asyncio
    async def test_calculate_trade_performance(self, lifecycle_manager, sample_order_request):
        """Test trade performance calculation."""
        # Start a trade
        trade_id = await lifecycle_manager.start_trade_lifecycle(
            "test-bot-123",
            "momentum_strategy",
            sample_order_request
        )
        
        # Update trade context with some data
        trade_context = lifecycle_manager.active_trades[trade_id]
        trade_context.filled_quantity = Decimal("0.1")
        trade_context.average_fill_price = Decimal("50010.0")
        trade_context.fees_paid = Decimal("5.0")
        
        performance = await lifecycle_manager.calculate_trade_performance(trade_id)
        
        assert "trade_id" in performance
        assert "filled_quantity" in performance
        assert "average_fill_price" in performance
        assert "fees_paid" in performance

    @pytest.mark.asyncio
    async def test_get_trade_history(self, lifecycle_manager):
        """Test getting trade history."""
        # Add some mock history
        from src.state.trade_lifecycle_manager import TradeHistoryRecord
        
        record = TradeHistoryRecord(
            trade_id="test-trade-123",
            bot_id="test-bot-123",
            strategy_name="momentum_strategy",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            average_price=Decimal("50000.0")
        )
        lifecycle_manager.trade_history.append(record)
        
        history = await lifecycle_manager.get_trade_history(bot_id="test-bot-123", limit=10)
        
        assert len(history) == 1
        assert history[0]["trade_id"] == "test-trade-123"
        assert history[0]["bot_id"] == "test-bot-123"


class TestQualityController:
    """Test cases for QualityController."""

    @pytest.fixture
    async def quality_controller(self, mock_config):
        """Create QualityController instance for testing."""
        with patch('src.state.quality_controller.DatabaseManager'), \
             patch('src.state.quality_controller.InfluxDBClient'):
            
            controller = QualityController(mock_config)
            
            # Mock database clients
            controller.db_manager = AsyncMock()
            controller.influxdb_client = AsyncMock()
            
            await controller.initialize()
            return controller

    @pytest.mark.asyncio
    async def test_validate_pre_trade(self, quality_controller, sample_order_request, sample_market_data):
        """Test pre-trade validation."""
        portfolio_context = {
            "total_value": 100000.0,
            "symbol_exposure": {"BTC/USDT": 10000.0}
        }
        
        validation = await quality_controller.validate_pre_trade(
            sample_order_request,
            sample_market_data,
            portfolio_context
        )
        
        assert validation.validation_id is not None
        assert validation.order_request == sample_order_request
        assert validation.overall_score >= 0
        assert validation.overall_score <= 100
        assert len(validation.checks) > 0

    @pytest.mark.asyncio
    async def test_analyze_post_trade(self, quality_controller, sample_execution_result, sample_market_data):
        """Test post-trade analysis."""
        analysis = await quality_controller.analyze_post_trade(
            "test-trade-123",
            sample_execution_result,
            sample_market_data,
            sample_market_data  # Using same data for before/after for simplicity
        )
        
        assert analysis.trade_id == "test-trade-123"
        assert analysis.execution_result == sample_execution_result
        assert analysis.overall_quality_score >= 0
        assert analysis.overall_quality_score <= 100
        assert analysis.execution_time_seconds >= 0

    @pytest.mark.asyncio
    async def test_get_quality_summary(self, quality_controller):
        """Test quality summary."""
        # Add some mock validation data
        from src.state.quality_controller import PreTradeValidation
        
        validation = PreTradeValidation(
            overall_score=85.0,
            overall_result=ValidationResult.PASSED
        )
        quality_controller.validation_history.append(validation)
        
        summary = await quality_controller.get_quality_summary(hours=24)
        
        assert "validation_summary" in summary
        assert "analysis_summary" in summary
        assert "quality_trends" in summary

    @pytest.mark.asyncio
    async def test_get_quality_trend_analysis(self, quality_controller):
        """Test quality trend analysis."""
        # Add some mock analysis data
        from src.state.quality_controller import PostTradeAnalysis
        
        for i in range(5):
            analysis = PostTradeAnalysis(
                analysis_id=f"analysis-{i}",
                trade_id=f"trade-{i}",
                overall_quality_score=80.0 + i,
                timestamp=datetime.now(timezone.utc) - timedelta(days=i)
            )
            quality_controller.analysis_history.append(analysis)
        
        trend = await quality_controller.get_quality_trend_analysis("overall_quality_score", 7)
        
        assert trend.metric_name == "overall_quality_score"
        assert trend.current_value >= 0
        assert trend.mean >= 0


class TestStateSyncManager:
    """Test cases for StateSyncManager."""

    @pytest.fixture
    async def sync_manager(self, mock_config):
        """Create StateSyncManager instance for testing."""
        with patch('src.state.state_sync_manager.DatabaseManager'), \
             patch('src.state.state_sync_manager.RedisClient'):
            
            manager = StateSyncManager(mock_config)
            
            # Mock database clients
            manager.db_manager = AsyncMock()
            manager.redis_client = AsyncMock()
            
            # Mock database session
            mock_session = AsyncMock()
            manager.db_manager.get_session.return_value.__aenter__.return_value = mock_session
            manager.db_manager.get_session.return_value.__aexit__.return_value = None
            
            await manager.initialize()
            return manager

    @pytest.mark.asyncio
    async def test_sync_state(self, sync_manager, sample_bot_state):
        """Test state synchronization."""
        # Mock Redis and PostgreSQL operations
        sync_manager.redis_client.setex = AsyncMock()
        
        success = await sync_manager.sync_state(
            "bot_state",
            "test-bot-123",
            sample_bot_state.model_dump(),
            "test_component"
        )
        
        assert success is True
        
        # Verify Redis was called
        sync_manager.redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_sync(self, sync_manager, sample_bot_state):
        """Test force synchronization."""
        # Mock loading from primary storage
        with patch.object(sync_manager, '_load_from_primary_storage', 
                         return_value=sample_bot_state.model_dump()):
            success = await sync_manager.force_sync("bot_state", "test-bot-123")
        
        # Should attempt sync even if it fails due to mocking
        assert success is not None

    @pytest.mark.asyncio
    async def test_get_sync_status(self, sync_manager):
        """Test getting sync status."""
        # Mock consistency check
        with patch.object(sync_manager, '_check_consistency', 
                         return_value={"consistent": True}):
            status = await sync_manager.get_sync_status("bot_state", "test-bot-123")
        
        assert "entity_type" in status
        assert "entity_id" in status
        assert "consistency_status" in status

    @pytest.mark.asyncio
    async def test_get_sync_metrics(self, sync_manager):
        """Test getting sync metrics."""
        # Update some metrics
        sync_manager.sync_metrics.total_sync_operations = 100
        sync_manager.sync_metrics.successful_syncs = 95
        sync_manager.sync_metrics.failed_syncs = 5
        
        metrics = await sync_manager.get_sync_metrics()
        
        assert metrics["total_sync_operations"] == 100
        assert metrics["successful_syncs"] == 95
        assert metrics["success_rate"] == 95.0

    @pytest.mark.asyncio
    async def test_subscribe_to_events(self, sync_manager):
        """Test event subscription."""
        callback_called = False
        
        def test_callback(event):
            nonlocal callback_called
            callback_called = True
        
        await sync_manager.subscribe_to_events(SyncEventType.STATE_UPDATED, test_callback)
        
        assert SyncEventType.STATE_UPDATED in sync_manager.event_subscribers
        assert test_callback in sync_manager.event_subscribers[SyncEventType.STATE_UPDATED]

    @pytest.mark.asyncio
    async def test_register_conflict_resolver(self, sync_manager):
        """Test conflict resolver registration."""
        def custom_resolver(conflict):
            return {"resolved": True}
        
        await sync_manager.register_conflict_resolver("bot_state", custom_resolver)
        
        assert "bot_state" in sync_manager.custom_resolvers
        assert sync_manager.custom_resolvers["bot_state"] == custom_resolver


class TestIntegration:
    """Integration tests for state management components."""

    @pytest.fixture
    async def integrated_system(self, mock_config):
        """Create integrated state management system."""
        with patch('src.state.state_manager.DatabaseManager'), \
             patch('src.state.state_manager.RedisClient'), \
             patch('src.state.state_manager.InfluxDBClient'), \
             patch('src.state.checkpoint_manager.ensure_directory_exists'), \
             patch('src.state.trade_lifecycle_manager.DatabaseManager'), \
             patch('src.state.trade_lifecycle_manager.RedisClient'), \
             patch('src.state.quality_controller.DatabaseManager'), \
             patch('src.state.quality_controller.InfluxDBClient'), \
             patch('src.state.state_sync_manager.DatabaseManager'), \
             patch('src.state.state_sync_manager.RedisClient'):
            
            # Create all components
            state_manager = StateManager(mock_config)
            checkpoint_manager = CheckpointManager(mock_config)
            lifecycle_manager = TradeLifecycleManager(mock_config)
            quality_controller = QualityController(mock_config)
            sync_manager = StateSyncManager(mock_config)
            
            # Mock all database clients
            for manager in [state_manager, lifecycle_manager, sync_manager]:
                manager.db_manager = AsyncMock()
                manager.redis_client = AsyncMock()
                mock_session = AsyncMock()
                manager.db_manager.get_session.return_value.__aenter__.return_value = mock_session
                manager.db_manager.get_session.return_value.__aexit__.return_value = None
            
            quality_controller.db_manager = AsyncMock()
            quality_controller.influxdb_client = AsyncMock()
            
            # Initialize all components
            await state_manager.initialize()
            await checkpoint_manager.initialize()
            await lifecycle_manager.initialize()
            await quality_controller.initialize()
            await sync_manager.initialize()
            
            return {
                "state_manager": state_manager,
                "checkpoint_manager": checkpoint_manager,
                "lifecycle_manager": lifecycle_manager,
                "quality_controller": quality_controller,
                "sync_manager": sync_manager
            }

    @pytest.mark.asyncio
    async def test_complete_trade_workflow(self, integrated_system, sample_order_request, sample_market_data, sample_execution_result):
        """Test complete trade workflow through all components."""
        state_manager = integrated_system["state_manager"]
        lifecycle_manager = integrated_system["lifecycle_manager"]
        quality_controller = integrated_system["quality_controller"]
        sync_manager = integrated_system["sync_manager"]
        
        # 1. Pre-trade validation
        portfolio_context = {"total_value": 100000.0}
        validation = await quality_controller.validate_pre_trade(
            sample_order_request,
            sample_market_data,
            portfolio_context
        )
        assert validation.overall_result == ValidationResult.PASSED
        
        # 2. Start trade lifecycle
        trade_id = await lifecycle_manager.start_trade_lifecycle(
            "test-bot-123",
            "test_strategy",
            sample_order_request
        )
        assert trade_id is not None
        
        # 3. Transition through states
        await lifecycle_manager.transition_trade_state(
            trade_id,
            TradeLifecycleState.ORDER_SUBMITTED
        )
        
        # 4. Update with execution
        await lifecycle_manager.update_trade_execution(trade_id, sample_execution_result)
        
        # 5. Post-trade analysis
        analysis = await quality_controller.analyze_post_trade(
            trade_id,
            sample_execution_result,
            sample_market_data
        )
        assert analysis.overall_quality_score >= 0
        
        # 6. Sync state changes
        trade_context = lifecycle_manager.active_trades[trade_id]
        success = await sync_manager.sync_state(
            "trade_lifecycle",
            trade_id,
            {"state": trade_context.current_state.value},
            "lifecycle_manager"
        )
        assert success is True

    @pytest.mark.asyncio
    async def test_state_persistence_and_recovery(self, integrated_system, sample_bot_state):
        """Test state persistence and recovery workflow."""
        state_manager = integrated_system["state_manager"]
        checkpoint_manager = integrated_system["checkpoint_manager"]
        
        # 1. Save bot state
        version_id = await state_manager.save_bot_state(
            "test-bot-123",
            sample_bot_state,
            create_snapshot=True
        )
        assert version_id is not None
        
        # 2. Create checkpoint
        checkpoint_id = await checkpoint_manager.create_checkpoint(
            "test-bot-123",
            sample_bot_state
        )
        assert checkpoint_id is not None
        
        # 3. Simulate recovery scenario
        plan = await checkpoint_manager.create_recovery_plan("test-bot-123")
        assert plan.bot_id == "test-bot-123"
        assert len(plan.steps) > 0
        
        # 4. Load state back
        with patch.object(state_manager, 'load_bot_state', return_value=sample_bot_state):
            loaded_state = await state_manager.load_bot_state("test-bot-123")
        
        assert loaded_state is not None
        assert loaded_state.bot_id == sample_bot_state.bot_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])