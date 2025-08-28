"""Unit tests for BotInstance component."""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import Config
from src.core.types import (
    BotConfiguration, BotStatus, BotType, BotPriority,
    OrderRequest, OrderSide, OrderType, MarketData
)
from src.core.types.strategy import StrategyType
from src.bot_management.bot_instance import BotInstance


@pytest.fixture
def config():
    """Create test configuration."""
    config = MagicMock(spec=Config)
    config.error_handling = MagicMock()
    config.bot_management = {
        "heartbeat_interval": 30,
        "position_timeout_minutes": 60,
        "max_restart_attempts": 3
    }
    return config


@pytest.fixture
def bot_config():
    """Create test bot configuration."""
    return BotConfiguration(
        bot_id="test_bot_001",
        name="Test Strategy Bot",  # Changed from bot_name
        bot_type=BotType.TRADING,
        version="1.0.0",  # Added required field
        strategy_id="momentum",  # Valid StrategyType
        exchanges=["binance"],
        symbols=["BTCUSDT"],
        max_capital=Decimal("10000"),  # Changed from allocated_capital
        max_position_size=Decimal("1000"),
        max_daily_loss=Decimal("200"),  # Added for risk management
        health_check_interval=30,  # Changed from heartbeat_interval
        auto_start=True,
        strategy_config={
            "risk_percentage": 0.02,
            "max_concurrent_positions": 3
        }  # Moved extra configs to strategy_config
    )


@pytest.fixture
def mock_execution_engine():
    """Create mock execution engine."""
    engine = AsyncMock()
    engine.execute_order = AsyncMock()
    engine.cancel_execution = AsyncMock(return_value=True)
    engine.get_engine_summary = AsyncMock(return_value={"test": "summary"})
    return engine


@pytest.fixture
def mock_strategy():
    """Create mock strategy."""
    strategy = AsyncMock()
    strategy.generate_signals = AsyncMock(return_value=[])
    strategy.start = AsyncMock()
    strategy.stop = AsyncMock()
    strategy.get_performance_metrics = AsyncMock(return_value={"test": "metrics"})
    return strategy


@pytest.fixture
def mock_exchange():
    """Create mock exchange."""
    exchange = AsyncMock()
    exchange.exchange_name = "binance"
    exchange.get_market_data = AsyncMock()
    exchange.place_order = AsyncMock()
    exchange.get_account_balance = AsyncMock(return_value=Decimal("10000"))
    return exchange


@pytest.fixture
def bot_instance(config, bot_config):
    """Create BotInstance for testing."""
    with patch('src.bot_management.bot_instance.StrategyFactory') as mock_sf:
        with patch('src.bot_management.bot_instance.ExchangeFactory') as mock_ef:
            with patch('src.bot_management.bot_instance.ExecutionEngine') as mock_ee:
                with patch('src.bot_management.bot_instance.RiskService') as mock_rs:
                    with patch('src.bot_management.bot_instance.CapitalAllocatorAdapter') as mock_ca:
                        # Mock the factories to return mock instances
                        mock_sf_instance = AsyncMock()
                        mock_sf_instance.get_supported_strategies = AsyncMock(return_value=[StrategyType.MOMENTUM])
                        mock_sf_instance.create_strategy = AsyncMock()
                        mock_sf.return_value = mock_sf_instance
                        
                        mock_ef_instance = AsyncMock()
                        mock_ef_instance.get_exchange = AsyncMock()
                        mock_ef.return_value = mock_ef_instance
                        
                        mock_ee.return_value = AsyncMock()
                        mock_rs.return_value = AsyncMock()
                        mock_ca.return_value = AsyncMock()
                        return BotInstance(config, bot_config)


class TestBotInstance:
    """Test cases for BotInstance class."""

    @pytest.mark.asyncio
    async def test_bot_instance_initialization(self, bot_instance, bot_config):
        """Test bot instance initialization."""
        assert bot_instance.bot_config == bot_config
        assert bot_instance.bot_state.status == BotStatus.INITIALIZING
        assert bot_instance.bot_state.bot_id == "test_bot_001"
        assert not bot_instance.is_running
        assert bot_instance.strategy is None
        assert bot_instance.primary_exchange is None

    @pytest.mark.asyncio
    async def test_start_bot_success(self, bot_instance, mock_execution_engine, 
                                   mock_strategy, mock_exchange):
        """Test successful bot startup."""
        # Setup the existing mocks in bot_instance to return our test mocks
        bot_instance.strategy_factory.create_strategy = AsyncMock(return_value=mock_strategy)
        bot_instance.exchange_factory.get_exchange = AsyncMock(return_value=mock_exchange)
        
        # Start bot
        await bot_instance.start()
        
        # Verify state
        assert bot_instance.bot_state.status == BotStatus.RUNNING
        assert bot_instance.is_running
        assert bot_instance.strategy is not None
        assert bot_instance.primary_exchange is not None

    @pytest.mark.asyncio
    async def test_start_bot_already_running(self, bot_instance):
        """Test starting bot that's already running."""
        bot_instance.bot_state.status = BotStatus.RUNNING
        bot_instance.is_running = True
        
        # Should handle gracefully
        await bot_instance.start()
        assert bot_instance.bot_state.status == BotStatus.RUNNING

    @pytest.mark.asyncio
    async def test_stop_bot_success(self, bot_instance, mock_execution_engine, 
                                  mock_strategy, mock_exchange):
        """Test successful bot stop."""
        # Setup running bot
        bot_instance.bot_state.status = BotStatus.RUNNING
        bot_instance.is_running = True
        # Create actual asyncio tasks for proper cancellation behavior
        import asyncio
        
        async def dummy_coro():
            try:
                await asyncio.sleep(10)  # Long sleep to allow cancellation
            except asyncio.CancelledError:
                pass  # Expected when cancelled
                
        bot_instance.strategy_task = asyncio.create_task(dummy_coro())
        bot_instance.heartbeat_task = asyncio.create_task(dummy_coro())
        bot_instance.strategy = mock_strategy
        bot_instance.execution_engine = mock_execution_engine
        
        # Stop bot
        await bot_instance.stop()
        
        # Verify state
        assert bot_instance.bot_state.status == BotStatus.STOPPED
        assert not bot_instance.is_running

    @pytest.mark.asyncio
    async def test_pause_resume_bot(self, bot_instance):
        """Test bot pause and resume functionality."""
        # Setup running bot
        bot_instance.bot_state.status = BotStatus.RUNNING
        bot_instance.is_running = True
        
        # Pause bot
        await bot_instance.pause()
        assert bot_instance.bot_state.status == BotStatus.PAUSED
        
        # Resume bot
        await bot_instance.resume()
        assert bot_instance.bot_state.status == BotStatus.RUNNING

    @pytest.mark.asyncio
    async def test_execute_trade_success(self, bot_instance, mock_execution_engine):
        """Test successful trade execution."""
        # Setup
        bot_instance.execution_engine = mock_execution_engine
        bot_instance.bot_state.status = BotStatus.RUNNING
        
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        
        # Mock successful execution
        from src.core.types import ExecutionResult, ExecutionStatus, ExecutionAlgorithm
        execution_result = ExecutionResult(
            instruction_id="exec_123",
            symbol=order_request.symbol,
            status=ExecutionStatus.COMPLETED,
            target_quantity=order_request.quantity,
            filled_quantity=order_request.quantity,
            remaining_quantity=Decimal("0"),
            average_price=Decimal("50000"),
            worst_price=Decimal("50100"),
            best_price=Decimal("49900"),
            expected_cost=Decimal("50000"),
            actual_cost=Decimal("50000"),
            slippage_bps=0.0,
            slippage_amount=Decimal("0"),
            fill_rate=1.0,
            execution_time=30,
            num_fills=1,
            num_orders=1,
            total_fees=Decimal("10"),
            maker_fees=Decimal("5"),
            taker_fees=Decimal("5"),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc)
        )
        mock_execution_engine.execute_order.return_value = execution_result
        
        # Execute trade
        result = await bot_instance.execute_trade(order_request, {})
        
        # Verify
        assert result is not None
        assert len(bot_instance.order_history) == 1
        assert bot_instance.performance_metrics["total_trades"] == 1

    @pytest.mark.asyncio
    async def test_execute_trade_paused_bot(self, bot_instance):
        """Test trade execution when bot is paused."""
        bot_instance.bot_state.status = BotStatus.PAUSED
        
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        
        # Should not execute when paused
        result = await bot_instance.execute_trade(order_request, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_update_position(self, bot_instance):
        """Test position update functionality."""
        position_data = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("1.0"),
            "average_price": Decimal("50000"),
            "unrealized_pnl": Decimal("100")
        }
        
        await bot_instance.update_position("BTCUSDT", position_data)
        
        assert "BTCUSDT" in bot_instance.active_positions
        assert bot_instance.active_positions["BTCUSDT"] == position_data

    @pytest.mark.asyncio
    async def test_close_position(self, bot_instance, mock_execution_engine):
        """Test position closure."""
        # Setup position
        bot_instance.active_positions["BTCUSDT"] = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("1.0"),
            "average_price": Decimal("50000")
        }
        bot_instance.execution_engine = mock_execution_engine
        
        # Mock successful execution
        from src.core.types import ExecutionResult, ExecutionStatus, ExecutionAlgorithm
        
        # Create a real OrderRequest instead of MagicMock
        close_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        
        execution_result = ExecutionResult(
            instruction_id="exec_close_123",
            symbol=close_order.symbol,
            status=ExecutionStatus.COMPLETED,
            target_quantity=close_order.quantity,
            filled_quantity=close_order.quantity,
            remaining_quantity=Decimal("0"),
            average_price=Decimal("50000"),
            worst_price=Decimal("50100"),
            best_price=Decimal("49900"),
            expected_cost=Decimal("50000"),
            actual_cost=Decimal("50000"),
            slippage_bps=0.0,
            slippage_amount=Decimal("0"),
            fill_rate=1.0,
            execution_time=30,
            num_fills=1,
            num_orders=1,
            total_fees=Decimal("10"),
            maker_fees=Decimal("5"),
            taker_fees=Decimal("5"),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc)
        )
        mock_execution_engine.execute_order.return_value = execution_result
        
        # Close position
        result = await bot_instance.close_position("BTCUSDT", "manual")
        
        # Verify
        assert result is True
        assert "BTCUSDT" not in bot_instance.active_positions

    @pytest.mark.asyncio
    async def test_get_bot_summary(self, bot_instance):
        """Test bot summary generation."""
        # Setup some state
        bot_instance.bot_state.status = BotStatus.RUNNING
        bot_instance.performance_metrics["total_trades"] = 5
        bot_instance.performance_metrics["profitable_trades"] = 3
        
        summary = await bot_instance.get_bot_summary()
        
        # Verify summary structure
        assert "bot_info" in summary
        assert "status" in summary
        assert "performance" in summary
        assert "positions" in summary
        assert "recent_activity" in summary
        
        # Verify content
        assert summary["bot_info"]["bot_id"] == bot_instance.bot_config.bot_id
        assert summary["status"]["current_status"] == BotStatus.RUNNING.value
        assert summary["performance"]["total_trades"] == 5

    @pytest.mark.asyncio
    async def test_heartbeat_functionality(self, bot_instance):
        """Test heartbeat generation."""
        bot_instance.bot_state.status = BotStatus.RUNNING
        bot_instance.bot_state.last_heartbeat = datetime.now(timezone.utc)
        
        heartbeat = await bot_instance.get_heartbeat()
        
        assert "bot_id" in heartbeat
        assert "status" in heartbeat
        assert "timestamp" in heartbeat
        assert "health_metrics" in heartbeat
        assert heartbeat["bot_id"] == bot_instance.bot_config.bot_id

    @pytest.mark.asyncio
    async def test_error_handling_in_trading_loop(self, bot_instance, mock_strategy):
        """Test error handling in trading loop."""
        # Setup
        bot_instance.strategy = mock_strategy
        bot_instance.bot_state.status = BotStatus.RUNNING
        
        # Mock strategy to raise exception
        mock_strategy.generate_signals.side_effect = Exception("Strategy error")
        
        # Should handle error gracefully
        await bot_instance._trading_loop()
        
        # Bot should still be running (error handled)
        assert bot_instance.bot_state.status == BotStatus.RUNNING

    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, bot_instance):
        """Test performance metrics calculation."""
        # Add some trade history
        bot_instance.order_history = [
            {"pnl": Decimal("100"), "timestamp": datetime.now(timezone.utc)},
            {"pnl": Decimal("-50"), "timestamp": datetime.now(timezone.utc)},
            {"pnl": Decimal("200"), "timestamp": datetime.now(timezone.utc)}
        ]
        
        await bot_instance._calculate_performance_metrics()
        
        metrics = bot_instance.performance_metrics
        assert metrics["total_trades"] == 3
        assert metrics["profitable_trades"] == 2
        assert metrics["total_pnl"] == Decimal("250")
        assert metrics["win_rate"] == 2/3

    @pytest.mark.asyncio
    async def test_risk_check_functionality(self, bot_instance):
        """Test risk checking before trade execution."""
        # Setup position limits
        bot_instance.active_positions = {
            "BTCUSDT": {"quantity": Decimal("1.0")},
            "ETHUSDT": {"quantity": Decimal("2.0")},
            "ADAUSDT": {"quantity": Decimal("1.5")}
        }
        
        order_request = OrderRequest(
            symbol="DOTUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        
        # Should reject due to max concurrent positions (3)
        can_trade = await bot_instance._check_risk_limits(order_request)
        assert not can_trade

    @pytest.mark.asyncio
    async def test_bot_restart_after_error(self, bot_instance):
        """Test bot restart functionality after error."""
        # Simulate error state
        bot_instance.bot_state.status = BotStatus.ERROR
        bot_instance.bot_metrics.error_count = 1
        
        # Restart bot
        with patch.object(bot_instance, 'start') as mock_start:
            await bot_instance.restart("error_recovery")
            mock_start.assert_called_once()