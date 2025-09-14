"""Strategy Repository with proper database integration.

This module provides data access layer for strategy operations,
following the repository pattern with actual database integration.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from src.core.exceptions import RepositoryError, ValidationError
from src.core.logging import get_logger
from src.core.types.strategy import StrategyMetrics
from src.database.models import (
    AnalyticsStrategyMetrics,
    Signal,
    StateSnapshot,
    Strategy,
    Trade,
)
from src.database.repository.base import DatabaseRepository
from src.utils.decimal_utils import to_decimal

logger = get_logger(__name__)


class StrategyRepositoryInterface(ABC):
    """Interface for strategy data repository operations."""

    @abstractmethod
    async def create_strategy(self, strategy: Strategy) -> Strategy:
        """Create a new strategy."""
        pass

    @abstractmethod
    async def get_strategy(self, strategy_id: str) -> Strategy | None:
        """Get strategy by ID."""
        pass

    @abstractmethod
    async def update_strategy(self, strategy_id: str, updates: dict[str, Any]) -> Strategy | None:
        """Update strategy."""
        pass

    @abstractmethod
    async def delete_strategy(self, strategy_id: str) -> bool:
        """Delete strategy."""
        pass

    @abstractmethod
    async def get_strategies_by_bot(self, bot_id: str) -> list[Strategy]:
        """Get all strategies for a bot."""
        pass

    @abstractmethod
    async def get_active_strategies(self, bot_id: str | None = None) -> list[Strategy]:
        """Get active strategies."""
        pass

    @abstractmethod
    async def save_strategy_state(self, strategy_id: str, state_data: dict[str, Any]) -> bool:
        """Save strategy state."""
        pass

    @abstractmethod
    async def load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None:
        """Load strategy state."""
        pass

    @abstractmethod
    async def save_strategy_metrics(self, strategy_id: str, metrics: StrategyMetrics) -> bool:
        """Save strategy performance metrics."""
        pass

    @abstractmethod
    async def get_strategy_metrics(
        self, strategy_id: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[AnalyticsStrategyMetrics]:
        """Get strategy performance metrics."""
        pass

    @abstractmethod
    async def save_strategy_signals(self, signals: list[Signal]) -> list[Signal]:
        """Save strategy signals."""
        pass

    @abstractmethod
    async def get_strategy_signals(
        self, strategy_id: str, limit: int | None = None
    ) -> list[Signal]:
        """Get strategy signals."""
        pass


class StrategyRepository(DatabaseRepository, StrategyRepositoryInterface):
    """Strategy repository with database integration using UoW pattern."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        super().__init__(
            session=session,
            model=Strategy,
            entity_type=Strategy,
            key_type=str,
            name="StrategyRepository",
        )

    async def create_strategy(self, strategy: Strategy) -> Strategy:
        """Create a new strategy."""
        try:
            self.session.add(strategy)
            await self.session.flush()
            await self.session.refresh(strategy)
            logger.info(f"Strategy created: {strategy.id}")
            return strategy
        except Exception as e:
            logger.error(f"Failed to create strategy: {e}")
            raise RepositoryError(f"Failed to create strategy: {e}") from e

    async def get_strategy(self, strategy_id: str) -> Strategy | None:
        """Get strategy by ID with all related data."""
        try:
            # Parse UUID if string
            strategy_uuid: UUID
            if isinstance(strategy_id, str):
                strategy_uuid = UUID(strategy_id)
            else:
                strategy_uuid = strategy_id

            stmt = (
                select(Strategy)
                .options(
                    joinedload(Strategy.bot),
                    selectinload(Strategy.orders),
                    selectinload(Strategy.positions),
                    selectinload(Strategy.trades),
                    selectinload(Strategy.signals),
                    selectinload(Strategy.strategy_metrics),
                )
                .where(Strategy.id == strategy_uuid)
            )

            result = await self.session.execute(stmt)
            strategy = result.unique().scalar_one_or_none()

            if strategy:
                logger.debug(f"Strategy found: {strategy.id}")
            else:
                logger.debug(f"Strategy not found: {strategy_id}")

            return strategy

        except Exception as e:
            logger.error(f"Failed to get strategy {strategy_id}: {e}")
            raise RepositoryError(f"Failed to get strategy: {e}") from e

    async def update_strategy(self, strategy_id: str, updates: dict[str, Any]) -> Strategy | None:
        """Update strategy."""
        try:
            strategy = await self.get_strategy(strategy_id)
            if not strategy:
                return None

            # Validate update fields
            allowed_fields = {
                "name",
                "status",
                "params",
                "max_position_size",
                "risk_per_trade",
                "stop_loss_percentage",
                "take_profit_percentage",
            }

            invalid_fields = set(updates.keys()) - allowed_fields
            if invalid_fields:
                raise ValidationError(f"Invalid update fields: {invalid_fields}")

            # Apply updates
            for field, value in updates.items():
                if field in [
                    "max_position_size",
                    "risk_per_trade",
                    "stop_loss_percentage",
                    "take_profit_percentage",
                ]:
                    value = to_decimal(value) if value is not None else None
                setattr(strategy, field, value)

            await self.session.flush()
            logger.info(f"Strategy updated: {strategy_id}")
            return strategy

        except Exception as e:
            logger.error(f"Failed to update strategy {strategy_id}: {e}")
            raise RepositoryError(f"Failed to update strategy: {e}") from e

    async def delete_strategy(self, strategy_id: str) -> bool:
        """Delete strategy."""
        try:
            strategy = await self.get_strategy(strategy_id)
            if not strategy:
                return False

            await self.session.delete(strategy)
            await self.session.flush()
            logger.info(f"Strategy deleted: {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete strategy {strategy_id}: {e}")
            raise RepositoryError(f"Failed to delete strategy: {e}") from e

    async def get_strategies_by_bot(self, bot_id: str) -> list[Strategy]:
        """Get all strategies for a bot."""
        try:
            bot_uuid: UUID
            if isinstance(bot_id, str):
                bot_uuid = UUID(bot_id)
            else:
                bot_uuid = bot_id

            stmt = (
                select(Strategy)
                .options(selectinload(Strategy.strategy_metrics))
                .where(Strategy.bot_id == bot_uuid)
                .order_by(Strategy.created_at.desc())
            )

            result = await self.session.execute(stmt)
            strategies = result.scalars().all()

            logger.debug(f"Found {len(strategies)} strategies for bot {bot_id}")
            return list(strategies)

        except Exception as e:
            logger.error(f"Failed to get strategies for bot {bot_id}: {e}")
            raise RepositoryError(f"Failed to get strategies for bot: {e}") from e

    async def get_active_strategies(self, bot_id: str | None = None) -> list[Strategy]:
        """Get active strategies."""
        try:
            stmt = (
                select(Strategy)
                .options(selectinload(Strategy.strategy_metrics))
                .where(Strategy.status == "active")
            )

            if bot_id:
                bot_uuid: UUID
                if isinstance(bot_id, str):
                    bot_uuid = UUID(bot_id)
                else:
                    bot_uuid = bot_id
                stmt = stmt.where(Strategy.bot_id == bot_uuid)

            result = await self.session.execute(stmt)
            strategies = result.scalars().all()

            logger.debug(f"Found {len(strategies)} active strategies")
            return list(strategies)

        except Exception as e:
            logger.error(f"Failed to get active strategies: {e}")
            raise RepositoryError(f"Failed to get active strategies: {e}") from e

    async def save_strategy_state(self, strategy_id: str, state_data: dict[str, Any]) -> bool:
        """Save strategy state to state management system."""
        try:
            strategy_uuid: UUID
            if isinstance(strategy_id, str):
                strategy_uuid = UUID(strategy_id)
            else:
                strategy_uuid = strategy_id

            # Create or update state snapshot
            from uuid import uuid4

            # For now, create a new snapshot each time (could be optimized to update latest)
            state_record = StateSnapshot(
                snapshot_id=uuid4(),
                name=f"strategy_{strategy_uuid}_state",
                description="Strategy state snapshot",
                snapshot_type="strategy_state",
                entity_type="strategy",
                entity_id=str(strategy_uuid),
                data=state_data,
            )
            self.session.add(state_record)

            await self.session.flush()
            logger.debug(f"Strategy state saved for {strategy_uuid}")
            return True

        except Exception as e:
            logger.error(f"Failed to save strategy state for {strategy_id}: {e}")
            return False

    async def load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None:
        """Load strategy state from state management system."""
        try:
            strategy_uuid: UUID
            if isinstance(strategy_id, str):
                strategy_uuid = UUID(strategy_id)
            else:
                strategy_uuid = strategy_id

            # Get the latest state snapshot for the strategy
            stmt = (
                select(StateSnapshot)
                .where(
                    and_(
                        StateSnapshot.entity_id == str(strategy_uuid),
                        StateSnapshot.snapshot_type == "strategy_state",
                        StateSnapshot.deleted_at.is_(None),  # Not soft deleted
                    )
                )
                .order_by(StateSnapshot.created_at.desc())
                .limit(1)
            )

            result = await self.session.execute(stmt)
            state_record = result.scalar_one_or_none()

            if state_record:
                logger.debug(f"Strategy state loaded for {strategy_uuid}")
                return state_record.data

            logger.debug(f"No strategy state found for {strategy_uuid}")
            return None

        except Exception as e:
            logger.error(f"Failed to load strategy state for {strategy_id}: {e}")
            return None

    async def save_strategy_metrics(self, strategy_id: str, metrics: StrategyMetrics) -> bool:
        """Save strategy performance metrics."""
        try:
            strategy_uuid: UUID
            if isinstance(strategy_id, str):
                strategy_uuid = UUID(strategy_id)
            else:
                strategy_uuid = strategy_id

            # Get strategy to get bot_id
            strategy = await self.get_strategy(str(strategy_uuid))
            if not strategy:
                raise RepositoryError(f"Strategy not found: {strategy_uuid}")

            analytics_metrics = AnalyticsStrategyMetrics(
                bot_id=strategy.bot_id,
                strategy_id=strategy_uuid,
                strategy_name=strategy.name,
                timestamp=datetime.now(timezone.utc),
                total_return=to_decimal(getattr(metrics, "total_return", 0)),
                total_trades=getattr(metrics, "total_trades", 0),
                winning_trades=getattr(metrics, "winning_trades", 0),
                losing_trades=getattr(metrics, "losing_trades", 0),
                win_rate=to_decimal(getattr(metrics, "win_rate", 0)),
                profit_factor=to_decimal(getattr(metrics, "profit_factor", 0)),
                sharpe_ratio=to_decimal(getattr(metrics, "sharpe_ratio", 0)),
                max_drawdown=to_decimal(getattr(metrics, "max_drawdown", 0)),
                average_win=to_decimal(getattr(metrics, "average_win", 0)),
                average_loss=to_decimal(getattr(metrics, "average_loss", 0)),
            )

            self.session.add(analytics_metrics)
            await self.session.flush()
            logger.debug(f"Strategy metrics saved for {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save strategy metrics for {strategy_id}: {e}")
            return False

    async def get_strategy_metrics(
        self, strategy_id: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[AnalyticsStrategyMetrics]:
        """Get strategy performance metrics."""
        try:
            strategy_uuid: UUID
            if isinstance(strategy_id, str):
                strategy_uuid = UUID(strategy_id)
            else:
                strategy_uuid = strategy_id

            stmt = (
                select(AnalyticsStrategyMetrics)
                .where(AnalyticsStrategyMetrics.strategy_id == strategy_uuid)
                .order_by(AnalyticsStrategyMetrics.timestamp.desc())
            )

            if start_time:
                stmt = stmt.where(AnalyticsStrategyMetrics.timestamp >= start_time)
            if end_time:
                stmt = stmt.where(AnalyticsStrategyMetrics.timestamp <= end_time)

            result = await self.session.execute(stmt)
            metrics = result.scalars().all()

            logger.debug(f"Found {len(metrics)} metrics records for strategy {strategy_id}")
            return list(metrics)

        except Exception as e:
            logger.error(f"Failed to get strategy metrics for {strategy_id}: {e}")
            raise RepositoryError(f"Failed to get strategy metrics: {e}") from e

    async def save_strategy_signals(self, signals: list[Signal]) -> list[Signal]:
        """Save strategy signals."""
        try:
            saved_signals = []
            for signal in signals:
                self.session.add(signal)
                saved_signals.append(signal)

            await self.session.flush()

            # Refresh to get IDs
            for signal in saved_signals:
                await self.session.refresh(signal)

            logger.debug(f"Saved {len(saved_signals)} signals")
            return saved_signals

        except Exception as e:
            logger.error(f"Failed to save signals: {e}")
            raise RepositoryError(f"Failed to save signals: {e}") from e

    async def get_strategy_signals(
        self, strategy_id: str, limit: int | None = None
    ) -> list[Signal]:
        """Get strategy signals."""
        try:
            strategy_uuid: UUID
            if isinstance(strategy_id, str):
                strategy_uuid = UUID(strategy_id)
            else:
                strategy_uuid = strategy_id

            stmt = (
                select(Signal)
                .where(Signal.strategy_id == strategy_uuid)
                .order_by(Signal.created_at.desc())
            )

            if limit:
                stmt = stmt.limit(limit)

            result = await self.session.execute(stmt)
            signals = result.scalars().all()

            logger.debug(f"Found {len(signals)} signals for strategy {strategy_id}")
            return list(signals)

        except Exception as e:
            logger.error(f"Failed to get strategy signals for {strategy_id}: {e}")
            raise RepositoryError(f"Failed to get strategy signals: {e}") from e

    async def get_strategy_trades(
        self, strategy_id: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[Trade]:
        """Get trades for a strategy."""
        try:
            strategy_uuid: UUID
            if isinstance(strategy_id, str):
                strategy_uuid = UUID(strategy_id)
            else:
                strategy_uuid = strategy_id

            stmt = (
                select(Trade)
                .where(Trade.strategy_id == strategy_uuid)
                .order_by(Trade.created_at.desc())
            )

            if start_time:
                stmt = stmt.where(Trade.created_at >= start_time)
            if end_time:
                stmt = stmt.where(Trade.created_at <= end_time)

            result = await self.session.execute(stmt)
            trades = result.scalars().all()

            logger.debug(f"Found {len(trades)} trades for strategy {strategy_id}")
            return list(trades)

        except Exception as e:
            logger.error(f"Failed to get strategy trades for {strategy_id}: {e}")
            raise RepositoryError(f"Failed to get strategy trades: {e}") from e

    async def get_strategy_performance_summary(self, strategy_id: str) -> dict[str, Any]:
        """Get strategy performance summary."""
        try:
            strategy_uuid: UUID
            if isinstance(strategy_id, str):
                strategy_uuid = UUID(strategy_id)
            else:
                strategy_uuid = strategy_id

            # Get latest metrics
            latest_metrics = await self.get_strategy_metrics(strategy_id)
            if not latest_metrics:
                return {"strategy_id": str(strategy_id), "error": "No metrics found"}

            latest = latest_metrics[0]

            # Get trades count and P&L
            trades = await self.get_strategy_trades(str(strategy_id))
            total_pnl = sum(trade.pnl or Decimal("0") for trade in trades)

            # Get signals count
            signals = await self.get_strategy_signals(str(strategy_id))

            summary = {
                "strategy_id": str(strategy_id),
                "total_return": latest.total_return or Decimal("0"),
                "total_trades": latest.total_trades or 0,
                "winning_trades": latest.winning_trades or 0,
                "losing_trades": latest.losing_trades or 0,
                "win_rate": latest.win_rate or Decimal("0"),
                "profit_factor": latest.profit_factor or Decimal("0"),
                "sharpe_ratio": latest.sharpe_ratio or Decimal("0"),
                "max_drawdown": latest.max_drawdown or Decimal("0"),
                "total_pnl": total_pnl,
                "total_signals": len(signals),
                "last_updated": latest.timestamp,
            }

            logger.debug(f"Generated performance summary for strategy {strategy_id}")
            return summary

        except Exception as e:
            logger.error(f"Failed to get performance summary for {strategy_id}: {e}")
            return {"strategy_id": str(strategy_id), "error": str(e)}
