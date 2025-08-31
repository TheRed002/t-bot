"""Bot-specific repository implementations."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
from src.database.models.bot import Bot, BotLog, Signal, Strategy
from src.database.repository.base import DatabaseRepository
from src.database.repository.utils import RepositoryUtils

logger = get_logger(__name__)


class BotRepository(DatabaseRepository):
    """Repository for Bot entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(session=session, model=Bot, entity_type=Bot, key_type=str, name="BotRepository")

    async def get_active_bots(self) -> list[Bot]:
        """Get all active bots."""
        return await self.get_all(filters={"status": ["RUNNING", "PAUSED"]}, order_by="name")

    async def get_running_bots(self) -> list[Bot]:
        """Get running bots."""
        return await self.get_all(filters={"status": "RUNNING"})

    async def get_bot_by_name(self, name: str) -> Bot | None:
        """Get bot by name."""
        return await self.get_by(name=name)

    async def start_bot(self, bot_id: str) -> bool:
        """Start a bot."""
        bot = await self.get(bot_id)
        if bot and bot.status in ("STOPPED", "PAUSED"):
            bot.status = "INITIALIZING"
            await self.update(bot)
            return True
        return False

    async def stop_bot(self, bot_id: str) -> bool:
        """Stop a bot."""
        bot = await self.get(bot_id)
        if bot and bot.status in ("RUNNING", "PAUSED", "ERROR"):
            bot.status = "STOPPING"
            await self.update(bot)
            return True
        return False

    async def pause_bot(self, bot_id: str) -> bool:
        """Pause a bot."""
        bot = await self.get(bot_id)
        if bot and bot.status == "RUNNING":
            bot.status = "PAUSED"
            await self.update(bot)
            return True
        return False

    async def update_bot_status(self, bot_id: str, status: str) -> bool:
        """Update bot status."""
        return await RepositoryUtils.update_entity_status(self, bot_id, status, "Bot")

    async def update_bot_metrics(self, bot_id: str, metrics: dict[str, Any]) -> bool:
        """Update bot performance metrics."""
        return await RepositoryUtils.update_entity_fields(self, bot_id, "Bot", **metrics)

    async def get_bot_performance(self, bot_id: str) -> dict[str, Any]:
        """Get bot performance metrics."""
        bot = await self.get(bot_id)
        if not bot:
            return {}

        return {
            "total_trades": bot.total_trades,
            "winning_trades": bot.winning_trades,
            "losing_trades": bot.losing_trades,
            "win_rate": bot.win_rate,
            "total_pnl": bot.total_pnl,
            "average_pnl": bot.average_pnl,
            "allocated_capital": bot.allocated_capital,
            "current_balance": bot.current_balance,
            "roi": (
                ((bot.current_balance - bot.allocated_capital) / bot.allocated_capital * 100)
                if bot.allocated_capital > 0
                else 0
            ),
        }


class StrategyRepository(DatabaseRepository):
    """Repository for Strategy entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(
            session=session,
            model=Strategy,
            entity_type=Strategy,
            key_type=str,
            name="StrategyRepository",
        )

    async def get_active_strategies(self, bot_id: str | None = None) -> list[Strategy]:
        """Get active strategies."""
        filters = {"status": "ACTIVE"}
        if bot_id:
            filters["bot_id"] = bot_id

        return await self.get_all(filters=filters)

    async def get_strategies_by_bot(self, bot_id: str) -> list[Strategy]:
        """Get all strategies for a bot."""
        return await RepositoryUtils.get_entities_by_field(self, "bot_id", bot_id)

    async def get_strategy_by_name(self, bot_id: str, name: str) -> Strategy | None:
        """Get strategy by name within a bot."""
        return await self.get_by(bot_id=bot_id, name=name)

    async def activate_strategy(self, strategy_id: str) -> bool:
        """Activate a strategy."""
        strategy = await self.get(strategy_id)
        if strategy and strategy.status in ("INACTIVE", "PAUSED"):
            return await RepositoryUtils.update_entity_status(self, strategy_id, "ACTIVE", "Strategy")
        return False

    async def deactivate_strategy(self, strategy_id: str) -> bool:
        """Deactivate a strategy."""
        strategy = await self.get(strategy_id)
        if strategy and strategy.status == "ACTIVE":
            return await RepositoryUtils.update_entity_status(self, strategy_id, "INACTIVE", "Strategy")
        return False

    async def update_strategy_params(self, strategy_id: str, params: dict[str, Any]) -> bool:
        """Update strategy parameters."""
        strategy = await self.get(strategy_id)
        if strategy:
            strategy.params.update(params)
            await self.update(strategy)
            return True
        return False

    async def update_strategy_metrics(self, strategy_id: str, metrics: dict[str, Any]) -> bool:
        """Update strategy performance metrics."""
        return await RepositoryUtils.update_entity_fields(self, strategy_id, "Strategy", **metrics)


class SignalRepository(DatabaseRepository):
    """Repository for Signal entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(session=session, model=Signal, entity_type=Signal, key_type=str, name="SignalRepository")

    async def get_unexecuted_signals(self, strategy_id: str | None = None) -> list[Signal]:
        """Get unexecuted signals."""
        filters = {"executed": False}
        if strategy_id:
            filters["strategy_id"] = strategy_id

        return await self.get_all(filters=filters, order_by="-created_at")

    async def get_signals_by_strategy(self, strategy_id: str, limit: int = 100) -> list[Signal]:
        """Get signals for a strategy."""
        return await self.get_all(filters={"strategy_id": strategy_id}, order_by="-created_at", limit=limit)

    async def get_recent_signals(self, hours: int = 24, strategy_id: str | None = None) -> list[Signal]:
        """Get recent signals."""
        additional_filters = {"strategy_id": strategy_id} if strategy_id else None
        return await RepositoryUtils.get_recent_entities(self, hours, additional_filters)

    async def mark_signal_executed(self, signal_id: str, order_id: str, execution_time: Decimal) -> bool:
        """Mark signal as executed."""
        signal = await self.get(signal_id)
        if signal:
            signal.executed = True
            signal.order_id = order_id
            signal.execution_time = execution_time
            await self.update(signal)
            return True
        return False

    async def update_signal_outcome(
        self,
        signal_id: str,
        outcome: str,
        pnl: Decimal | None = None,
    ) -> bool:
        """Update signal outcome."""
        signal = await self.get(signal_id)
        if signal:
            signal.outcome = outcome
            if pnl is not None:
                signal.pnl = pnl
            await self.update(signal)
            return True
        return False

    async def get_signal_statistics(self, strategy_id: str, since: datetime | None = None) -> dict[str, Any]:
        """Get signal statistics for a strategy."""
        from sqlalchemy import select

        stmt = select(Signal).where(Signal.strategy_id == strategy_id)

        if since:
            stmt = stmt.where(Signal.created_at >= since)

        result = await self.session.execute(stmt)
        signals = list(result.scalars().all())

        if not signals:
            return {
                "total_signals": 0,
                "executed_signals": 0,
                "successful_signals": 0,
                "execution_rate": 0,
                "success_rate": 0,
                "average_execution_time": 0,
            }

        executed = [s for s in signals if s.executed]
        successful = [s for s in executed if s.outcome == "SUCCESS"]
        execution_times = [s.execution_time for s in executed if s.execution_time]

        return {
            "total_signals": len(signals),
            "executed_signals": len(executed),
            "successful_signals": len(successful),
            "execution_rate": (len(executed) / len(signals)) * 100,
            "success_rate": (len(successful) / len(executed)) * 100 if executed else 0,
            "average_execution_time": (sum(execution_times) / len(execution_times) if execution_times else 0),
        }


class BotLogRepository(DatabaseRepository):
    """Repository for BotLog entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(session=session, model=BotLog, entity_type=BotLog, key_type=str, name="BotLogRepository")

    async def get_logs_by_bot(self, bot_id: str, level: str | None = None, limit: int = 100) -> list[BotLog]:
        """Get logs for a bot."""
        filters = {"bot_id": bot_id}
        if level:
            filters["level"] = level

        return await self.get_all(filters=filters, order_by="-created_at", limit=limit)

    async def get_error_logs(self, bot_id: str | None = None, hours: int = 24) -> list[BotLog]:
        """Get error logs."""
        filters = {"level": ["ERROR", "CRITICAL"]}
        if bot_id:
            filters["bot_id"] = bot_id
        return await self._execute_recent_query(timestamp_field="created_at", hours=hours, additional_filters=filters)

    async def log_event(
        self,
        bot_id: str,
        level: str,
        message: str,
        category: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> BotLog:
        """Log an event."""
        log = BotLog(bot_id=bot_id, level=level, message=message, category=category, context=context)

        return await self.create(log)

    async def cleanup_old_logs(self, days: int = 30) -> int:
        """Delete logs older than specified days."""
        from datetime import timedelta

        from sqlalchemy import delete

        cutoff_time = datetime.now().replace(tzinfo=None) - timedelta(days=days)
        stmt = delete(self.model).where(self.model.created_at < cutoff_time)

        result = await self.session.execute(stmt)
        await self.session.flush()
        return result.rowcount

    async def _execute_recent_query(
        self,
        timestamp_field: str,
        hours: int,
        additional_filters: dict[str, Any] | None = None,
    ) -> list[BotLog]:
        """Execute query for recent entities within time range."""
        from datetime import timedelta

        from sqlalchemy import select

        cutoff_time = datetime.now().replace(tzinfo=None) - timedelta(hours=hours)
        stmt = select(self.model).where(getattr(self.model, timestamp_field) >= cutoff_time)

        if additional_filters:
            for key, value in additional_filters.items():
                if hasattr(self.model, key):
                    column = getattr(self.model, key)
                    if isinstance(value, list):
                        stmt = stmt.where(column.in_(value))
                    else:
                        stmt = stmt.where(column == value)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def _execute_cleanup_query(self, timestamp_field: str, days: int) -> list[BotLog]:
        """Execute query to get entities for cleanup."""
        from datetime import timedelta

        from sqlalchemy import select

        cutoff_time = datetime.now().replace(tzinfo=None) - timedelta(days=days)
        stmt = select(self.model).where(getattr(self.model, timestamp_field) < cutoff_time)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())
