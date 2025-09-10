"""
Unit tests for bot repository implementations.

This module tests all bot-related repositories including BotRepository,
StrategyRepository, SignalRepository, and BotLogRepository.
"""

import uuid
from datetime import datetime, timezone

# Removed decimal import as Bot model uses Float not Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import RepositoryError
from src.database.models.bot import Bot, BotLog, Signal, Strategy
from src.database.repository.bot import (
    BotLogRepository,
    BotRepository,
    SignalRepository,
    StrategyRepository,
)


class TestBotRepository:
    """Test BotRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        session = AsyncMock(spec=AsyncSession)
        # Make sync methods regular mocks to avoid warnings
        session.delete = Mock()
        session.add = Mock()
        session.merge = AsyncMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def bot_repository(self, mock_session):
        """Create BotRepository instance for testing."""
        return BotRepository(mock_session)

    @pytest.fixture
    def sample_bot(self):
        """Create sample bot entity."""
        return Bot(
            id=str(uuid.uuid4()),
            name="test_bot",
            description="Test bot description",
            status="RUNNING",
            exchange="binance",
            allocated_capital=10000.00,
            current_balance=10500.00,
            total_trades=15,
            winning_trades=10,
            losing_trades=5,
            total_pnl=500.00,
        )

    def test_bot_repository_init(self, mock_session):
        """Test BotRepository initialization."""
        repo = BotRepository(mock_session)

        assert repo.session == mock_session
        assert repo.model == Bot
        assert repo.name == "BotRepository"

    @pytest.mark.asyncio
    async def test_get_active_bots(self, bot_repository, mock_session):
        """Test get_active_bots method."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(status="RUNNING"), Mock(status="PAUSED")]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await bot_repository.get_active_bots()

        assert len(result) == 2
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_running_bots(self, bot_repository, mock_session):
        """Test get_running_bots method."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(status="RUNNING")]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await bot_repository.get_running_bots()

        assert len(result) == 1
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_bot_by_name_found(self, bot_repository, mock_session, sample_bot):
        """Test get_bot_by_name when bot exists."""
        with patch.object(bot_repository, "get_by", return_value=sample_bot):
            result = await bot_repository.get_bot_by_name("test_bot")

            assert result == sample_bot

    @pytest.mark.asyncio
    async def test_get_bot_by_name_not_found(self, bot_repository):
        """Test get_bot_by_name when bot doesn't exist."""
        with patch.object(bot_repository, "get_by", return_value=None):
            result = await bot_repository.get_bot_by_name("nonexistent_bot")

            assert result is None

    @pytest.mark.asyncio
    async def test_start_bot_success(self, bot_repository, sample_bot):
        """Test successful bot start operation."""
        sample_bot.status = "STOPPED"

        with (
            patch.object(bot_repository, "get", return_value=sample_bot),
            patch.object(bot_repository, "update", return_value=sample_bot),
        ):
            result = await bot_repository.start_bot(sample_bot.id)

            assert result is True
            assert sample_bot.status == "INITIALIZING"

    @pytest.mark.asyncio
    async def test_start_bot_invalid_status(self, bot_repository, sample_bot):
        """Test start bot with invalid current status."""
        sample_bot.status = "RUNNING"  # Already running

        with patch.object(bot_repository, "get", return_value=sample_bot):
            result = await bot_repository.start_bot(sample_bot.id)

            assert result is False

    @pytest.mark.asyncio
    async def test_start_bot_not_found(self, bot_repository):
        """Test start bot when bot doesn't exist."""
        with patch.object(bot_repository, "get", return_value=None):
            result = await bot_repository.start_bot("nonexistent_id")

            assert result is False

    @pytest.mark.asyncio
    async def test_stop_bot_success(self, bot_repository, sample_bot):
        """Test successful bot stop operation."""
        sample_bot.status = "RUNNING"

        with (
            patch.object(bot_repository, "get", return_value=sample_bot),
            patch.object(bot_repository, "update", return_value=sample_bot),
        ):
            result = await bot_repository.stop_bot(sample_bot.id)

            assert result is True
            assert sample_bot.status == "STOPPING"

    @pytest.mark.asyncio
    async def test_stop_bot_invalid_status(self, bot_repository, sample_bot):
        """Test stop bot with invalid current status."""
        sample_bot.status = "STOPPED"  # Already stopped

        with patch.object(bot_repository, "get", return_value=sample_bot):
            result = await bot_repository.stop_bot(sample_bot.id)

            assert result is False

    @pytest.mark.asyncio
    async def test_pause_bot_success(self, bot_repository, sample_bot):
        """Test successful bot pause operation."""
        sample_bot.status = "RUNNING"

        with (
            patch.object(bot_repository, "get", return_value=sample_bot),
            patch.object(bot_repository, "update", return_value=sample_bot),
        ):
            result = await bot_repository.pause_bot(sample_bot.id)

            assert result is True
            assert sample_bot.status == "PAUSED"

    @pytest.mark.asyncio
    async def test_pause_bot_invalid_status(self, bot_repository, sample_bot):
        """Test pause bot with invalid current status."""
        sample_bot.status = "STOPPED"

        with patch.object(bot_repository, "get", return_value=sample_bot):
            result = await bot_repository.pause_bot(sample_bot.id)

            assert result is False

    @pytest.mark.asyncio
    async def test_update_bot_status_success(self, bot_repository, sample_bot):
        """Test successful bot status update."""
        with (
            patch.object(bot_repository, "get", return_value=sample_bot),
            patch.object(bot_repository, "update", return_value=sample_bot),
        ):
            result = await bot_repository.update_bot_status(sample_bot.id, "ERROR")

            assert result is True
            assert sample_bot.status == "ERROR"

    @pytest.mark.asyncio
    async def test_update_bot_status_bot_not_found(self, bot_repository):
        """Test update bot status when bot doesn't exist."""
        with patch.object(bot_repository, "get", return_value=None):
            result = await bot_repository.update_bot_status("nonexistent_id", "ERROR")

            assert result is False

    @pytest.mark.asyncio
    async def test_update_bot_metrics_success(self, bot_repository, sample_bot):
        """Test successful bot metrics update."""
        metrics = {"total_trades": 20, "winning_trades": 12, "total_pnl": 600.00}

        with (
            patch.object(bot_repository, "get", return_value=sample_bot),
            patch.object(bot_repository, "update", return_value=sample_bot),
        ):
            result = await bot_repository.update_bot_metrics(sample_bot.id, metrics)

            assert result is True
            assert sample_bot.total_trades == 20
            assert sample_bot.winning_trades == 12
            assert sample_bot.total_pnl == 600.00

    @pytest.mark.asyncio
    async def test_update_bot_metrics_invalid_attribute(self, bot_repository, sample_bot):
        """Test update bot metrics with invalid attribute."""
        metrics = {"invalid_attribute": "value", "total_trades": 20}

        with (
            patch.object(bot_repository, "get", return_value=sample_bot),
            patch.object(bot_repository, "update", return_value=sample_bot),
        ):
            result = await bot_repository.update_bot_metrics(sample_bot.id, metrics)

            assert result is True
            assert sample_bot.total_trades == 20
            # Invalid attribute should be ignored

    @pytest.mark.asyncio
    async def test_get_bot_performance(self, bot_repository, sample_bot):
        """Test get bot performance metrics."""
        with patch.object(bot_repository, "get", return_value=sample_bot):
            result = await bot_repository.get_bot_performance(sample_bot.id)

            expected = {
                "total_trades": sample_bot.total_trades,
                "winning_trades": sample_bot.winning_trades,
                "losing_trades": sample_bot.losing_trades,
                "win_rate": sample_bot.win_rate,
                "total_pnl": sample_bot.total_pnl,
                "average_pnl": sample_bot.average_pnl,
                "allocated_capital": sample_bot.allocated_capital,
                "current_balance": sample_bot.current_balance,
                "roi": (
                    (sample_bot.current_balance - sample_bot.allocated_capital)
                    / sample_bot.allocated_capital
                    * 100
                ),
            }

            assert result == expected

    @pytest.mark.asyncio
    async def test_get_bot_performance_zero_allocated_capital(self, bot_repository, sample_bot):
        """Test get bot performance with zero allocated capital."""
        sample_bot.allocated_capital = 0.0

        with patch.object(bot_repository, "get", return_value=sample_bot):
            result = await bot_repository.get_bot_performance(sample_bot.id)

            assert result["roi"] == 0

    @pytest.mark.asyncio
    async def test_get_bot_performance_bot_not_found(self, bot_repository):
        """Test get bot performance when bot doesn't exist."""
        with patch.object(bot_repository, "get", return_value=None):
            result = await bot_repository.get_bot_performance("nonexistent_id")

            assert result == {}


class TestStrategyRepository:
    """Test StrategyRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def strategy_repository(self, mock_session):
        """Create StrategyRepository instance for testing."""
        return StrategyRepository(mock_session)

    @pytest.fixture
    def sample_strategy(self):
        """Create sample strategy entity."""
        return Strategy(
            id=str(uuid.uuid4()),
            name="test_strategy",
            bot_id=str(uuid.uuid4()),
            type="trend_following",
            status="ACTIVE",
            params={"param1": "value1"},
            max_position_size=1000.0,
            risk_per_trade=0.02,
            total_signals=15,
            executed_signals=10,
            successful_signals=6,
        )

    @pytest.mark.asyncio
    async def test_get_active_strategies_no_bot_filter(self, strategy_repository, mock_session):
        """Test get active strategies without bot filter."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(status="ACTIVE"), Mock(status="ACTIVE")]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await strategy_repository.get_active_strategies()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_active_strategies_with_bot_filter(self, strategy_repository, mock_session):
        """Test get active strategies with bot filter."""
        bot_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(status="ACTIVE", bot_id=bot_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await strategy_repository.get_active_strategies(bot_id=bot_id)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_strategies_by_bot(self, strategy_repository, mock_session):
        """Test get strategies by bot."""
        bot_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(bot_id=bot_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await strategy_repository.get_strategies_by_bot(bot_id)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_strategy_by_name(self, strategy_repository, sample_strategy):
        """Test get strategy by name."""
        with patch.object(strategy_repository, "get_by", return_value=sample_strategy):
            result = await strategy_repository.get_strategy_by_name(
                sample_strategy.bot_id, sample_strategy.name
            )

            assert result == sample_strategy

    @pytest.mark.asyncio
    async def test_activate_strategy_success(self, strategy_repository, sample_strategy):
        """Test successful strategy activation."""
        sample_strategy.status = "INACTIVE"

        with (
            patch.object(strategy_repository, "get", return_value=sample_strategy),
            patch.object(strategy_repository, "update", return_value=sample_strategy),
        ):
            result = await strategy_repository.activate_strategy(sample_strategy.id)

            assert result is True
            assert sample_strategy.status == "ACTIVE"

    @pytest.mark.asyncio
    async def test_activate_strategy_invalid_status(self, strategy_repository, sample_strategy):
        """Test activate strategy with invalid current status."""
        sample_strategy.status = "ACTIVE"  # Already active

        with patch.object(strategy_repository, "get", return_value=sample_strategy):
            result = await strategy_repository.activate_strategy(sample_strategy.id)

            assert result is False

    @pytest.mark.asyncio
    async def test_deactivate_strategy_success(self, strategy_repository, sample_strategy):
        """Test successful strategy deactivation."""
        sample_strategy.status = "ACTIVE"

        with (
            patch.object(strategy_repository, "get", return_value=sample_strategy),
            patch.object(strategy_repository, "update", return_value=sample_strategy),
        ):
            result = await strategy_repository.deactivate_strategy(sample_strategy.id)

            assert result is True
            assert sample_strategy.status == "INACTIVE"

    @pytest.mark.asyncio
    async def test_update_strategy_params_success(self, strategy_repository, sample_strategy):
        """Test successful strategy parameters update."""
        new_params = {"param1": "new_value", "param2": "value2"}

        with (
            patch.object(strategy_repository, "get", return_value=sample_strategy),
            patch.object(strategy_repository, "update", return_value=sample_strategy),
        ):
            result = await strategy_repository.update_strategy_params(
                sample_strategy.id, new_params
            )

            assert result is True
            assert sample_strategy.params["param1"] == "new_value"
            assert sample_strategy.params["param2"] == "value2"

    @pytest.mark.asyncio
    async def test_update_strategy_metrics_success(self, strategy_repository, sample_strategy):
        """Test successful strategy metrics update."""
        metrics = {"total_signals": 15, "executed_signals": 9}

        with (
            patch.object(strategy_repository, "get", return_value=sample_strategy),
            patch.object(strategy_repository, "update", return_value=sample_strategy),
        ):
            result = await strategy_repository.update_strategy_metrics(sample_strategy.id, metrics)

            assert result is True
            assert sample_strategy.total_signals == 15
            assert sample_strategy.executed_signals == 9


class TestSignalRepository:
    """Test SignalRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def signal_repository(self, mock_session):
        """Create SignalRepository instance for testing."""
        return SignalRepository(mock_session)

    @pytest.fixture
    def sample_signal(self):
        """Create sample signal entity."""
        return Signal(
            id=str(uuid.uuid4()),
            strategy_id=str(uuid.uuid4()),
            action="BUY",
            symbol="BTCUSD",
            strength=0.8,
            executed=False,
            created_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_get_unexecuted_signals_no_strategy_filter(self, signal_repository, mock_session):
        """Test get unexecuted signals without strategy filter."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(executed=False), Mock(executed=False)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await signal_repository.get_unexecuted_signals()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_unexecuted_signals_with_strategy_filter(
        self, signal_repository, mock_session
    ):
        """Test get unexecuted signals with strategy filter."""
        strategy_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(executed=False, strategy_id=strategy_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await signal_repository.get_unexecuted_signals(strategy_id=strategy_id)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_signals_by_strategy(self, signal_repository, mock_session):
        """Test get signals by strategy."""
        strategy_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(strategy_id=strategy_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await signal_repository.get_signals_by_strategy(strategy_id, limit=50)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_recent_signals(self, signal_repository, mock_session):
        """Test get recent signals."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(created_at=datetime.now(timezone.utc))]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await signal_repository.get_recent_signals(hours=12)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_mark_signal_executed_success(self, signal_repository, sample_signal):
        """Test successful mark signal as executed."""
        order_id = str(uuid.uuid4())
        execution_time = 1.5

        with (
            patch.object(signal_repository, "get", return_value=sample_signal),
            patch.object(signal_repository, "update", return_value=sample_signal),
        ):
            result = await signal_repository.mark_signal_executed(
                sample_signal.id, order_id, execution_time
            )

            assert result is True
            assert sample_signal.executed is True
            assert sample_signal.order_id == order_id
            assert sample_signal.execution_time == execution_time

    @pytest.mark.asyncio
    async def test_mark_signal_executed_signal_not_found(self, signal_repository):
        """Test mark signal executed when signal doesn't exist."""
        with patch.object(signal_repository, "get", return_value=None):
            result = await signal_repository.mark_signal_executed("nonexistent_id", "order_id", 1.0)

            assert result is False

    @pytest.mark.asyncio
    async def test_update_signal_outcome_success(self, signal_repository, sample_signal):
        """Test successful signal outcome update."""
        outcome = "SUCCESS"
        pnl = 125.50

        with (
            patch.object(signal_repository, "get", return_value=sample_signal),
            patch.object(signal_repository, "update", return_value=sample_signal),
        ):
            result = await signal_repository.update_signal_outcome(sample_signal.id, outcome, pnl)

            assert result is True
            assert sample_signal.outcome == outcome
            assert sample_signal.pnl == pnl

    @pytest.mark.asyncio
    async def test_update_signal_outcome_without_pnl(self, signal_repository, sample_signal):
        """Test signal outcome update without PnL."""
        outcome = "FAILED"

        with (
            patch.object(signal_repository, "get", return_value=sample_signal),
            patch.object(signal_repository, "update", return_value=sample_signal),
        ):
            result = await signal_repository.update_signal_outcome(sample_signal.id, outcome)

            assert result is True
            assert sample_signal.outcome == outcome

    @pytest.mark.asyncio
    async def test_get_signal_statistics_with_signals(self, signal_repository, mock_session):
        """Test get signal statistics with signals data."""
        strategy_id = str(uuid.uuid4())

        # Mock signals with different outcomes
        signals = [
            Mock(executed=True, outcome="SUCCESS", execution_time=1.0),
            Mock(executed=True, outcome="SUCCESS", execution_time=2.0),
            Mock(executed=True, outcome="FAILED", execution_time=1.5),
            Mock(executed=False, outcome=None, execution_time=None),
        ]

        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = signals
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await signal_repository.get_signal_statistics(strategy_id)

        expected = {
            "total_signals": 4,
            "executed_signals": 3,
            "successful_signals": 2,
            "execution_rate": 75.0,
            "success_rate": 66.67,  # 2/3 * 100
            "average_execution_time": 1.5,  # (1.0 + 2.0 + 1.5) / 3
        }

        assert result["total_signals"] == expected["total_signals"]
        assert result["executed_signals"] == expected["executed_signals"]
        assert result["successful_signals"] == expected["successful_signals"]
        assert result["execution_rate"] == expected["execution_rate"]

    @pytest.mark.asyncio
    async def test_get_signal_statistics_no_signals(self, signal_repository, mock_session):
        """Test get signal statistics with no signals."""
        strategy_id = str(uuid.uuid4())

        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await signal_repository.get_signal_statistics(strategy_id)

        expected = {
            "total_signals": 0,
            "executed_signals": 0,
            "successful_signals": 0,
            "execution_rate": 0,
            "success_rate": 0,
            "average_execution_time": 0,
        }

        assert result == expected


class TestBotLogRepository:
    """Test BotLogRepository implementation."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def log_repository(self, mock_session):
        """Create BotLogRepository instance for testing."""
        return BotLogRepository(mock_session)

    @pytest.fixture
    def sample_log(self):
        """Create sample log entity."""
        return BotLog(
            id=str(uuid.uuid4()),
            bot_id=str(uuid.uuid4()),
            level="INFO",
            message="Test log message",
            category="GENERAL",
            context={"key": "value"},
            created_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_get_logs_by_bot_no_level_filter(self, log_repository, mock_session):
        """Test get logs by bot without level filter."""
        bot_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(bot_id=bot_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await log_repository.get_logs_by_bot(bot_id)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_logs_by_bot_with_level_filter(self, log_repository, mock_session):
        """Test get logs by bot with level filter."""
        bot_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(bot_id=bot_id, level="ERROR")]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await log_repository.get_logs_by_bot(bot_id, level="ERROR", limit=50)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_error_logs_no_bot_filter(self, log_repository, mock_session):
        """Test get error logs without bot filter."""
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(level="ERROR"), Mock(level="CRITICAL")]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await log_repository.get_error_logs()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_error_logs_with_bot_filter(self, log_repository, mock_session):
        """Test get error logs with bot filter."""
        bot_id = str(uuid.uuid4())
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = [Mock(level="ERROR", bot_id=bot_id)]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await log_repository.get_error_logs(bot_id=bot_id, hours=12)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_log_event_success(self, log_repository):
        """Test successful log event creation."""
        bot_id = str(uuid.uuid4())
        level = "INFO"
        message = "Test message"
        category = "TRADING"
        context = {"symbol": "BTCUSD"}

        mock_log = BotLog(
            bot_id=bot_id, level=level, message=message, category=category, context=context
        )

        with patch.object(log_repository, "create", return_value=mock_log):
            result = await log_repository.log_event(bot_id, level, message, category, context)

            assert result == mock_log
            assert result.bot_id == bot_id
            assert result.level == level
            assert result.message == message

    @pytest.mark.asyncio
    async def test_log_event_minimal_params(self, log_repository):
        """Test log event with minimal parameters."""
        bot_id = str(uuid.uuid4())
        level = "ERROR"
        message = "Error occurred"

        mock_log = BotLog(bot_id=bot_id, level=level, message=message, category=None, context=None)

        with patch.object(log_repository, "create", return_value=mock_log):
            result = await log_repository.log_event(bot_id, level, message)

            assert result == mock_log
            assert result.category is None
            assert result.context is None

    @pytest.mark.asyncio
    async def test_cleanup_old_logs_success(self, log_repository, mock_session):
        """Test successful cleanup of old logs."""
        # Mock delete operation
        mock_result = Mock()
        mock_result.rowcount = 15
        mock_session.execute.return_value = mock_result

        result = await log_repository.cleanup_old_logs(days=30)

        assert result == 15
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_old_logs_no_logs(self, log_repository, mock_session):
        """Test cleanup when no old logs exist."""
        mock_result = Mock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        result = await log_repository.cleanup_old_logs(days=7)

        assert result == 0


class TestBotRepositoryErrorHandling:
    """Test error handling in bot repositories."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def bot_repository(self, mock_session):
        """Create BotRepository instance for testing."""
        return BotRepository(mock_session)

    @pytest.mark.asyncio
    async def test_database_error_handling(self, bot_repository, mock_session):
        """Test database error handling in repository operations."""
        mock_session.execute.side_effect = SQLAlchemyError("Database connection lost")

        with pytest.raises(RepositoryError):
            await bot_repository.get_active_bots()

    @pytest.mark.asyncio
    async def test_integrity_error_handling(self, bot_repository, mock_session):
        """Test integrity error handling during create operations."""
        bot = Bot(id=str(uuid.uuid4()), name="test_bot", status="RUNNING")
        mock_session.flush.side_effect = IntegrityError("Duplicate key", None, None)

        with pytest.raises(RepositoryError):
            await bot_repository.create(bot)

        mock_session.rollback.assert_called_once()


class TestBotRepositoryConcurrency:
    """Test concurrent operations in bot repositories."""

    @pytest.fixture
    def mock_session(self):
        """Create mock AsyncSession for testing."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def bot_repository(self, mock_session):
        """Create BotRepository instance for testing."""
        return BotRepository(mock_session)

    @pytest.mark.asyncio
    async def test_concurrent_bot_status_updates(self, bot_repository):
        """Test concurrent bot status updates."""
        bot_id = str(uuid.uuid4())
        bot = Bot(id=bot_id, name="test_bot", status="RUNNING")

        with (
            patch.object(bot_repository, "get", return_value=bot),
            patch.object(bot_repository, "update", return_value=bot),
        ):
            # Simulate concurrent updates
            result1 = await bot_repository.update_bot_status(bot_id, "PAUSED")
            result2 = await bot_repository.update_bot_status(bot_id, "STOPPED")

            assert result1 is True
            assert result2 is True

    @pytest.mark.asyncio
    async def test_concurrent_metric_updates(self, bot_repository):
        """Test concurrent metric updates."""
        bot_id = str(uuid.uuid4())
        bot = Bot(id=bot_id, name="test_bot", status="RUNNING", total_trades=10)

        with (
            patch.object(bot_repository, "get", return_value=bot),
            patch.object(bot_repository, "update", return_value=bot),
        ):
            # Simulate concurrent metric updates
            metrics1 = {"total_trades": 15}
            metrics2 = {"winning_trades": 8}

            result1 = await bot_repository.update_bot_metrics(bot_id, metrics1)
            result2 = await bot_repository.update_bot_metrics(bot_id, metrics2)

            assert result1 is True
            assert result2 is True
