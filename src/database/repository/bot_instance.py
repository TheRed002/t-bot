"""
Bot Instance Repository - Data access layer for bot instances.

This repository handles all database operations for bot instances,
providing centralized data access for bot lifecycle management.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.bot_instance import BotInstance
from src.database.repository.base import DatabaseRepository


class BotInstanceRepository(DatabaseRepository):
    """Repository for bot instance operations."""

    def __init__(self, session: AsyncSession):
        """
        Initialize bot instance repository.

        Args:
            session: Database session
        """
        super().__init__(
            session=session,
            model=BotInstance,
            entity_type=BotInstance,
            key_type=str,
            name="BotInstanceRepository",
        )

    async def get_by_bot_id(self, bot_id: str) -> BotInstance | None:
        """
        Get bot instance by bot ID.

        Args:
            bot_id: Unique bot identifier

        Returns:
            Bot instance or None if not found
        """
        result = await self.session.execute(select(BotInstance).where(BotInstance.bot_id == bot_id))
        return result.scalar_one_or_none()

    async def get_active_bots(self) -> list[BotInstance]:
        """
        Get all active bot instances.

        Returns:
            List of active bot instances
        """
        result = await self.session.execute(
            select(BotInstance)
            .where(
                or_(
                    BotInstance.status == "running",
                    BotInstance.status == "starting",
                )
            )
            .order_by(BotInstance.created_at)
        )
        return list(result.scalars().all())

    async def get_bots_by_status(self, status: str) -> list[BotInstance]:
        """
        Get bot instances by status.

        Args:
            status: Bot status (running, stopped, error, etc.)

        Returns:
            List of bot instances with specified status
        """
        result = await self.session.execute(
            select(BotInstance).where(BotInstance.status == status).order_by(desc(BotInstance.updated_at))
        )
        return list(result.scalars().all())

    async def get_bots_by_strategy(self, strategy_name: str) -> list[BotInstance]:
        """
        Get bot instances using a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            List of bot instances using the strategy
        """
        result = await self.session.execute(
            select(BotInstance).where(BotInstance.strategy_name == strategy_name).order_by(BotInstance.created_at)
        )
        return list(result.scalars().all())

    async def get_bots_by_exchange(self, exchange: str) -> list[BotInstance]:
        """
        Get bot instances for a specific exchange.

        Args:
            exchange: Exchange name

        Returns:
            List of bot instances for the exchange
        """
        result = await self.session.execute(
            select(BotInstance).where(BotInstance.exchange == exchange).order_by(BotInstance.created_at)
        )
        return list(result.scalars().all())

    async def update_bot_status(
        self,
        bot_id: str,
        status: str,
        error_message: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> BotInstance | None:
        """
        Update bot instance status and optionally error message and metrics.

        Args:
            bot_id: Bot identifier
            status: New status
            error_message: Optional error message
            metrics: Optional metrics to update

        Returns:
            Updated bot instance or None
        """
        bot = await self.get_by_bot_id(bot_id)
        if bot:
            bot.status = status
            if error_message:
                bot.error_message = error_message
                bot.error_count = (bot.error_count or 0) + 1
            if metrics:
                bot.performance_metrics = metrics
            bot.last_heartbeat = datetime.now(timezone.utc)
            bot.updated_at = datetime.now(timezone.utc)
            await self.session.commit()
            await self.session.refresh(bot)
        return bot

    async def update_heartbeat(self, bot_id: str) -> bool:
        """
        Update bot heartbeat timestamp.

        Args:
            bot_id: Bot identifier

        Returns:
            True if updated, False if bot not found
        """
        bot = await self.get_by_bot_id(bot_id)
        if bot:
            bot.last_heartbeat = datetime.now(timezone.utc)
            await self.session.commit()
            return True
        return False

    async def get_stale_bots(self, heartbeat_timeout_seconds: int = 300) -> list[BotInstance]:
        """
        Get bots with stale heartbeats.

        Args:
            heartbeat_timeout_seconds: Timeout in seconds

        Returns:
            List of stale bot instances
        """
        from datetime import timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=heartbeat_timeout_seconds)

        result = await self.session.execute(
            select(BotInstance)
            .where(
                and_(
                    BotInstance.status.in_(["running", "starting"]),
                    or_(
                        BotInstance.last_heartbeat < cutoff_time,
                        BotInstance.last_heartbeat.is_(None),
                    ),
                )
            )
            .order_by(BotInstance.last_heartbeat)
        )
        return list(result.scalars().all())

    async def get_error_bots(self, min_error_count: int = 1) -> list[BotInstance]:
        """
        Get bots with errors.

        Args:
            min_error_count: Minimum error count

        Returns:
            List of bot instances with errors
        """
        result = await self.session.execute(
            select(BotInstance)
            .where(
                or_(
                    BotInstance.status == "error",
                    and_(
                        BotInstance.error_count.isnot(None),
                        BotInstance.error_count >= min_error_count,
                    ),
                )
            )
            .order_by(desc(BotInstance.error_count))
        )
        return list(result.scalars().all())

    async def reset_bot_errors(self, bot_id: str) -> bool:
        """
        Reset error count and message for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            True if reset, False if bot not found
        """
        bot = await self.get_by_bot_id(bot_id)
        if bot:
            bot.error_count = 0
            bot.error_message = None
            bot.updated_at = datetime.now(timezone.utc)
            await self.session.commit()
            return True
        return False

    async def get_bot_statistics(self) -> dict[str, Any]:
        """
        Get overall bot statistics.

        Returns:
            Dictionary with bot statistics
        """
        # Count bots by status
        status_counts = await self.session.execute(
            select(
                BotInstance.status,
                func.count(BotInstance.id).label("count"),
            ).group_by(BotInstance.status)
        )

        # Count bots by strategy
        strategy_counts = await self.session.execute(
            select(
                BotInstance.strategy_name,
                func.count(BotInstance.id).label("count"),
            ).group_by(BotInstance.strategy_name)
        )

        # Count bots by exchange
        exchange_counts = await self.session.execute(
            select(
                BotInstance.exchange,
                func.count(BotInstance.id).label("count"),
            ).group_by(BotInstance.exchange)
        )

        # Get error statistics
        error_stats = await self.session.execute(
            select(
                func.count(BotInstance.id).label("total_with_errors"),
                func.sum(BotInstance.error_count).label("total_errors"),
                func.avg(BotInstance.error_count).label("avg_errors"),
            ).where(BotInstance.error_count > 0)
        )

        status_dict = {row.status: row.count for row in status_counts}
        strategy_dict = {row.strategy_name: row.count for row in strategy_counts}
        exchange_dict = {row.exchange: row.count for row in exchange_counts}
        error_row = error_stats.first()

        return {
            "total_bots": sum(status_dict.values()),
            "by_status": status_dict,
            "by_strategy": strategy_dict,
            "by_exchange": exchange_dict,
            "error_statistics": {
                "bots_with_errors": error_row.total_with_errors or 0 if error_row else 0,
                "total_errors": int(error_row.total_errors or 0) if error_row else 0,
                "avg_errors_per_bot": str(error_row.avg_errors or 0) if error_row else "0",
            },
        }

    async def cleanup_old_bots(self, days: int = 30, status: str = "stopped") -> int:
        """
        Clean up old bot instances.

        Args:
            days: Number of days to keep
            status: Status of bots to clean

        Returns:
            Number of bots deleted
        """
        from datetime import timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)

        # Find bots to delete
        result = await self.session.execute(
            select(BotInstance).where(
                and_(
                    BotInstance.status == status,
                    BotInstance.updated_at < cutoff_time,
                )
            )
        )
        bots_to_delete = result.scalars().all()

        # Delete bots
        for bot in bots_to_delete:
            await self.session.delete(bot)

        await self.session.commit()
        return len(bots_to_delete)

    async def get_bots_by_symbol(self, symbol: str) -> list[BotInstance]:
        """
        Get bot instances trading a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of bot instances trading the symbol
        """
        result = await self.session.execute(
            select(BotInstance).where(BotInstance.symbol == symbol).order_by(BotInstance.created_at)
        )
        return list(result.scalars().all())

    async def get_profitable_bots(self, min_pnl: Decimal = Decimal("0.0")) -> list[BotInstance]:
        """
        Get profitable bot instances.

        Args:
            min_pnl: Minimum P&L threshold

        Returns:
            List of profitable bot instances
        """
        result = await self.session.execute(
            select(BotInstance)
            .where(
                and_(
                    BotInstance.total_pnl.isnot(None),
                    BotInstance.total_pnl > min_pnl,
                )
            )
            .order_by(desc(BotInstance.total_pnl))
        )
        return list(result.scalars().all())

    async def update_bot_metrics(
        self,
        bot_id: str,
        total_trades: int | None = None,
        total_pnl: Decimal | None = None,
        win_rate: Decimal | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> BotInstance | None:
        """
        Update bot performance metrics.

        Args:
            bot_id: Bot identifier
            total_trades: Total number of trades
            total_pnl: Total profit/loss
            win_rate: Win rate percentage
            metrics: Additional metrics

        Returns:
            Updated bot instance or None
        """
        bot = await self.get_by_bot_id(bot_id)
        if bot:
            if total_trades is not None:
                bot.total_trades = total_trades
            if total_pnl is not None:
                bot.total_pnl = total_pnl
            if win_rate is not None:
                bot.win_rate = win_rate
            if metrics:
                bot.performance_metrics = metrics
            bot.updated_at = datetime.now(timezone.utc)
            await self.session.commit()
            await self.session.refresh(bot)
        return bot
